"""AI API balance monitoring plugin for AstrBot."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path


DEFAULT_ALERT_MESSAGE = (
    "[余额告警]\n{account_name} 当前余额为 {balance} {unit}，"
    "已低于告警阈值 {threshold} {unit}。\n请及时充值，避免服务中断。"
)
DEFAULT_RECOVERY_MESSAGE = (
    "[余额恢复]\n{account_name} 当前余额已恢复至 {balance} {unit}，"
    "高于告警阈值 {threshold} {unit}。\n感谢您的支持！"
)
DEEPSEEK_BALANCE_URL = "https://api.deepseek.com/user/balance"
DEEPSEEK_CURRENCY = "CNY"
DEROUTER_BASE_URL = "https://cf-api.derouter.ai"
DEROUTER_ACCOUNT_ENDPOINT = "/balance"
DEROUTER_CUSTOMER_ENDPOINT = "/sub-key/balance"
DEROUTER_CURRENCY = "USD"
REQUEST_TIMEOUT_SECONDS = 15


class BalanceQueryError(RuntimeError):
    """A user-actionable error while reading a provider balance."""


@register(
    "ineedmoney",
    "rog",
    "定时检查 AI API 余额，并在余额不足时主动发送充值提醒。",
    "1.0.1",
)
class INeedMoneyPlugin(Star):
    """Monitor the balance of a supported AI provider and notify allowlisted chats."""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context, config)
        self.config = config
        self._polling_task: asyncio.Task | None = None
        self._check_lock = asyncio.Lock()
        self._last_alert_at: dict[str, float] = {}
        self._last_balances: dict[str, Decimal] = {}
        self._last_query_errors: dict[str, str] = {}
        self._low_balance_states: dict[str, bool] = {}

    async def initialize(self) -> None:
        """Start one lifecycle-managed task; it rereads WebUI settings each cycle."""
        self._migrate_legacy_config()
        self._polling_task = asyncio.create_task(
            self._poll_loop(), name="ineedmoney-balance-monitor"
        )
        logger.info("INeedMoney balance monitor initialized.")

    async def terminate(self) -> None:
        """Stop the polling task during plugin reload, disable, or shutdown."""
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None

    @filter.command("余额查询")
    async def query_balance(self, event: AstrMessageEvent):
        """管理员：查询所有已启用平台的余额，并验证接口配置。"""
        if not event.is_admin():
            yield event.plain_result("此命令仅 AstrBot 管理员可用。")
            return
        providers = self._enabled_providers()
        if not providers:
            yield event.plain_result(self._no_provider_text())
            return

        threshold = self._decimal_config("low_balance_threshold")
        sections: list[str] = []
        for provider in providers:
            try:
                balance = await self._query_provider_balance(provider)
            except BalanceQueryError as exc:
                self._last_query_errors[provider] = str(exc)
                sections.append(
                    f"{self._provider_label(provider)}：查询失败（{exc}）"
                )
                continue
            self._last_balances[provider] = balance
            self._last_query_errors.pop(provider, None)
            sections.append(self._provider_status_text(provider, balance, threshold))
        yield event.plain_result("\n\n".join(sections))

    @filter.command("余额告警测试")
    async def test_alert_in_chat(self, event: AstrMessageEvent):
        """管理员：使用示例低余额在当前会话测试各平台的告警样式。"""
        if not event.is_admin():
            yield event.plain_result("此命令仅 AstrBot 管理员可用。")
            return
        try:
            threshold = self._decimal_config("low_balance_threshold")
            if threshold < 0:
                raise BalanceQueryError("配置项 low_balance_threshold 不能小于 0")
            test_threshold = threshold if threshold > 0 else Decimal("1")
            test_balance = test_threshold / Decimal("2")
        except BalanceQueryError as exc:
            yield event.plain_result(f"余额告警测试失败：{exc}")
            return

        for provider in self._preview_providers():
            chain = MessageChain().message(
                f"[告警样式测试 · {self._provider_label(provider)}]\n"
                + await self._alert_text(
                    provider, test_balance, test_threshold, event.unified_msg_origin
                )
            )
            self._append_receipt_images(chain)
            yield event.chain_result(chain)

    @filter.command("余额恢复测试")
    async def test_recovery_in_chat(self, event: AstrMessageEvent):
        """管理员：使用示例高余额在当前会话测试各平台的恢复感谢消息。"""
        if not event.is_admin():
            yield event.plain_result("此命令仅 AstrBot 管理员可用。")
            return
        try:
            threshold = self._decimal_config("low_balance_threshold")
            if threshold < 0:
                raise BalanceQueryError("配置项 low_balance_threshold 不能小于 0")
            test_threshold = threshold
            test_balance = threshold + (Decimal("1") if threshold >= 0 else Decimal("2"))
        except BalanceQueryError as exc:
            yield event.plain_result(f"余额恢复测试失败：{exc}")
            return

        for provider in self._preview_providers():
            yield event.plain_result(
                f"[余额恢复测试 · {self._provider_label(provider)}]\n"
                + await self._recovery_text(
                    provider, test_balance, test_threshold, event.unified_msg_origin
                )
            )

    @filter.command("余额监控会话")
    async def show_session_id(self, event: AstrMessageEvent):
        """管理员：显示此群聊或私聊对应的主动消息会话 ID。"""
        if not event.is_admin():
            yield event.plain_result("此命令仅 AstrBot 管理员可用。")
            return
        yield event.plain_result(
            "将下面的会话标识复制到插件配置的“允许接收提醒的会话列表”中：\n"
            f"{event.unified_msg_origin}"
        )

    @filter.command("余额监控会话添加")
    async def add_notification_session(self, event: AstrMessageEvent):
        """管理员：将当前群聊或私聊加入余额提醒会话列表。"""
        if not event.is_admin():
            yield event.plain_result("此命令仅 AstrBot 管理员可用。")
            return
        session = event.unified_msg_origin.strip()
        sessions = self._string_list_config("notification_sessions")
        if session not in sessions:
            sessions.append(session)
            self.config["notification_sessions"] = sessions
            self.config.save_config()
            yield event.plain_result("当前会话已加入余额提醒列表。")
        else:
            yield event.plain_result("当前会话已经在余额提醒列表中。")

    @filter.command("余额监控状态")
    async def monitor_status(self, event: AstrMessageEvent):
        """管理员：查看各平台最近一次查询结果，不额外发起 API 请求。"""
        if not event.is_admin():
            yield event.plain_result("此命令仅 AstrBot 管理员可用。")
            return
        providers = self._enabled_providers()
        if not providers:
            yield event.plain_result(self._no_provider_text())
            return

        threshold = self._decimal_config("low_balance_threshold")
        sections: list[str] = []
        for provider in providers:
            error = self._last_query_errors.get(provider)
            balance = self._last_balances.get(provider)
            if error:
                sections.append(
                    f"{self._provider_label(provider)}：最近一次查询失败"
                    f"（{error}）"
                )
            elif balance is None:
                sections.append(
                    f"{self._provider_label(provider)}：尚未完成一次查询"
                )
            else:
                sections.append(
                    self._provider_status_text(provider, balance, threshold)
                )
        yield event.plain_result("\n\n".join(sections))

    async def _poll_loop(self) -> None:
        while True:
            try:
                if self._enabled():
                    await self._check_and_notify()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected error in INeedMoney balance monitor.")
            delay = self._poll_interval_seconds() if self._enabled() else 60
            await asyncio.sleep(delay)

    async def _check_and_notify(self) -> None:
        async with self._check_lock:
            providers = self._enabled_providers()
            if not providers:
                logger.warning(
                    "Balance monitor is enabled, but no provider switch is turned on."
                )
                return
            try:
                threshold = self._decimal_config("low_balance_threshold")
                if threshold < 0:
                    raise BalanceQueryError("配置项 low_balance_threshold 不能小于 0")
            except BalanceQueryError as exc:
                logger.warning("Balance monitor configuration is invalid: %s", exc)
                return

            for provider in providers:
                label = self._provider_label(provider)
                try:
                    balance = await self._query_provider_balance(provider)
                except BalanceQueryError as exc:
                    self._last_query_errors[provider] = str(exc)
                    logger.warning("[%s] Balance query failed: %s", label, exc)
                    continue
                self._last_balances[provider] = balance
                self._last_query_errors.pop(provider, None)

                if balance >= threshold:
                    if self._low_balance_states.get(provider):
                        logger.info(
                            "[%s] Balance recovered above the configured threshold.",
                            label,
                        )
                        await self._send_recovery_notice(
                            provider, balance, threshold
                        )
                    self._low_balance_states[provider] = False
                    self._last_alert_at.pop(provider, None)
                    continue

                self._low_balance_states[provider] = True
                if not self._alert_is_due(provider):
                    continue
                if await self._send_low_balance_alert(provider, balance, threshold):
                    self._last_alert_at[provider] = time.monotonic()

    async def _send_recovery_notice(
        self, provider: str, balance: Decimal, threshold: Decimal
    ) -> bool:
        sessions = self._string_list_config("notification_sessions")
        if not sessions:
            logger.warning(
                "Balance recovered, but no notification sessions are configured."
            )
            return False

        delivered = False
        for session in sessions:
            chain = MessageChain().message(
                await self._recovery_text(provider, balance, threshold, session)
            )
            try:
                sent = await self.context.send_message(session, chain)
                delivered = delivered or sent
                if not sent:
                    logger.warning(
                        "Recovery notice was not delivered to session %s.", session
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Failed to send recovery notice to session %s.", session
                )
        return delivered

    async def _query_provider_balance(self, provider: str) -> Decimal:
        if provider == "deepseek":
            return await self._query_deepseek_balance()
        if provider == "derouter":
            return await self._query_derouter_balance()
        if provider == "custom":
            return await self._query_custom_balance()
        raise BalanceQueryError(f"暂不支持的余额查询平台：{provider}")

    async def _query_deepseek_balance(self) -> Decimal:
        """Call the fixed DeepSeek endpoint so users only need to provide an API key."""
        api_key = self._deepseek_api_key()
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    DEEPSEEK_BALANCE_URL,
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                ) as response:
                    response_text = await response.text()
                    if not 200 <= response.status < 300:
                        raise BalanceQueryError(
                            f"DeepSeek 余额接口返回 HTTP {response.status}"
                        )
        except asyncio.CancelledError:
            raise
        except BalanceQueryError:
            raise
        except asyncio.TimeoutError as exc:
            raise BalanceQueryError("DeepSeek 余额接口请求超时") from exc
        except aiohttp.ClientError as exc:
            raise BalanceQueryError(f"无法连接 DeepSeek 余额接口：{exc}") from exc

        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise BalanceQueryError("DeepSeek 余额接口未返回 JSON 数据") from exc
        if not isinstance(payload, Mapping):
            raise BalanceQueryError("DeepSeek 余额接口返回的数据格式无效")
        if payload.get("is_available") is False:
            raise BalanceQueryError("DeepSeek 账户当前不可用，请检查账户状态")

        balance_infos = payload.get("balance_infos")
        if not isinstance(balance_infos, list):
            raise BalanceQueryError("DeepSeek 响应中没有余额信息")
        balance_info = next(
            (
                item
                for item in balance_infos
                if isinstance(item, Mapping)
                and str(item.get("currency", "")).upper() == DEEPSEEK_CURRENCY
            ),
            None,
        )
        if balance_info is None:
            raise BalanceQueryError(f"DeepSeek 响应中没有 {DEEPSEEK_CURRENCY} 余额")

        value = balance_info.get("total_balance")
        if value is None or isinstance(value, bool):
            raise BalanceQueryError("DeepSeek 返回的余额不是数值")
        try:
            balance = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise BalanceQueryError("DeepSeek 返回的余额不是可识别的数值") from exc
        if not balance.is_finite():
            raise BalanceQueryError("DeepSeek 返回的余额不是有限数值")
        return balance

    def _deepseek_api_key(self) -> str:
        """Return a bare DeepSeek key, accepting a pasted Bearer form."""
        api_key = self._string_config("deepseek_api_key").strip()
        if api_key.lower().startswith("bearer "):
            api_key = api_key[7:].strip()
        if not api_key:
            raise BalanceQueryError("请先在插件配置中填写 DeepSeek API Key")
        if "\r" in api_key or "\n" in api_key:
            raise BalanceQueryError("DeepSeek API Key 格式无效")
        return api_key

    async def _query_derouter_balance(self) -> Decimal:
        """Call derouter's preset endpoint; users only need to provide a key."""
        api_key = self._derouter_api_key()
        base_url = (
            self._string_config("derouter_base_url", DEROUTER_BASE_URL)
            .strip()
            .rstrip("/")
        )
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise BalanceQueryError("derouter API 地址必须是有效的 http 或 https URL")

        is_customer_key = self._derouter_key_type() == "customer"
        endpoint = (
            DEROUTER_CUSTOMER_ENDPOINT if is_customer_key else DEROUTER_ACCOUNT_ENDPOINT
        )
        field = "remaining" if is_customer_key else "available"
        key_label = "客户密钥" if is_customer_key else "账户密钥"
        api_label = f"derouter {key_label}余额接口"

        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    f"{base_url}{endpoint}",
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                ) as response:
                    response_text = await response.text()
                    if response.status == 401:
                        raise BalanceQueryError(f"{api_label} 鉴权失败，请检查密钥")
                    if not 200 <= response.status < 300:
                        raise BalanceQueryError(
                            f"{api_label} 返回 HTTP {response.status}"
                        )
        except asyncio.CancelledError:
            raise
        except BalanceQueryError:
            raise
        except asyncio.TimeoutError as exc:
            raise BalanceQueryError(f"{api_label} 请求超时") from exc
        except aiohttp.ClientError as exc:
            raise BalanceQueryError(f"无法连接 {api_label}：{exc}") from exc

        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise BalanceQueryError(f"{api_label} 未返回 JSON 数据") from exc
        if not isinstance(payload, Mapping):
            raise BalanceQueryError(f"{api_label} 返回的数据格式无效")
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            raise BalanceQueryError(f"{api_label} 返回错误：{error.strip()}")
        return self._to_balance(payload.get(field), f"{api_label} 的 {field} 字段")

    async def _query_custom_balance(self) -> Decimal:
        """Query an advanced, user-supplied JSON balance endpoint."""
        url = self._string_config("custom_balance_api_url").strip()
        if not url:
            raise BalanceQueryError("请先在自定义接口配置中填写余额 API 地址")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise BalanceQueryError("自定义余额 API 地址必须是有效的 http 或 https URL")

        method = self._string_config("custom_balance_api_method", "GET").strip().upper()
        if method not in {"GET", "POST"}:
            raise BalanceQueryError("自定义余额 API 请求方法仅支持 GET 或 POST")
        request_kwargs: dict[str, Any] = {"headers": self._build_custom_headers()}
        if method == "POST":
            request_kwargs["json"] = self._custom_request_json_body()

        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(method, url, **request_kwargs) as response:
                    response_text = await response.text()
                    if not 200 <= response.status < 300:
                        raise BalanceQueryError(
                            f"自定义余额 API 返回 HTTP {response.status}"
                        )
        except asyncio.CancelledError:
            raise
        except BalanceQueryError:
            raise
        except asyncio.TimeoutError as exc:
            raise BalanceQueryError("自定义余额 API 请求超时") from exc
        except aiohttp.ClientError as exc:
            raise BalanceQueryError(f"无法连接自定义余额 API：{exc}") from exc

        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise BalanceQueryError("自定义余额 API 未返回 JSON 数据") from exc
        value = self._extract_json_path(
            payload, self._string_config("custom_balance_json_path", "data.balance")
        )
        if value is None or isinstance(value, bool):
            raise BalanceQueryError("自定义接口返回的余额不是数值")
        try:
            balance = Decimal(str(value)) * self._decimal_config(
                "custom_balance_scale", "1"
            )
        except (InvalidOperation, ValueError) as exc:
            raise BalanceQueryError("自定义接口返回的余额不是可识别的数值") from exc
        if not balance.is_finite():
            raise BalanceQueryError("自定义接口返回的余额不是有限数值")
        return balance

    def _build_custom_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        token = self._string_config("custom_api_token").strip()
        if token:
            name = (
                self._string_config("custom_token_header_name", "Authorization").strip()
                or "Authorization"
            )
            self._validate_header(name, token)
            headers[name] = token

        custom_headers_text = self._string_config("custom_headers_json")
        if not custom_headers_text.strip():
            return headers
        try:
            custom_headers = json.loads(custom_headers_text)
        except json.JSONDecodeError as exc:
            raise BalanceQueryError("自定义请求头不是合法 JSON") from exc
        if not isinstance(custom_headers, Mapping):
            raise BalanceQueryError("自定义请求头必须是 JSON 对象")
        for name, value in custom_headers.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise BalanceQueryError("自定义请求头的名称和值都必须是字符串")
            self._validate_header(name, value)
            headers[name] = value
        return headers

    @staticmethod
    def _validate_header(name: str, value: str) -> None:
        if not name.strip() or "\r" in name or "\n" in name:
            raise BalanceQueryError("请求头名称不合法")
        if "\r" in value or "\n" in value:
            raise BalanceQueryError("请求头值不合法")

    def _custom_request_json_body(self) -> Any:
        body = self._string_config("custom_balance_api_body_json")
        if not body.strip():
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise BalanceQueryError("自定义请求体不是合法 JSON") from exc

    @staticmethod
    def _extract_json_path(payload: Any, path: str) -> Any:
        """Read dotted object keys and numeric indexes, e.g. data.users[0].balance."""
        path = path.strip().removeprefix("$").lstrip(".")
        if not path:
            raise BalanceQueryError("自定义余额 JSON 路径不能为空")
        current = payload
        for segment in path.split("."):
            match = re.fullmatch(r"([^\[\]]+)((?:\[\d+\])*)", segment)
            if not match:
                raise BalanceQueryError(f"自定义余额 JSON 路径格式无效：{path}")
            key, indexes = match.groups()
            if not isinstance(current, Mapping) or key not in current:
                raise BalanceQueryError(f"响应中找不到余额字段：{path}")
            current = current[key]
            for index in re.findall(r"\[(\d+)\]", indexes):
                if not isinstance(current, list) or int(index) >= len(current):
                    raise BalanceQueryError(f"响应中找不到余额字段：{path}")
                current = current[int(index)]
        return current

    async def _send_low_balance_alert(
        self, provider: str, balance: Decimal, threshold: Decimal
    ) -> bool:
        sessions = self._string_list_config("notification_sessions")
        if not sessions:
            logger.warning("Balance is low, but no notification sessions are configured.")
            return False
        delivered = False
        for session in sessions:
            chain = MessageChain().message(
                await self._alert_text(provider, balance, threshold, session)
            )
            self._append_receipt_images(chain)
            try:
                sent = await self.context.send_message(session, chain)
                delivered = delivered or sent
                if not sent:
                    logger.warning("Balance alert was not delivered to session %s.", session)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed to send balance alert to session %s.", session)
        return delivered

    async def _alert_text(
        self,
        provider: str,
        balance: Decimal,
        threshold: Decimal,
        session: str | None = None,
    ) -> str:
        fields = self._template_fields(provider, balance, threshold)
        template, persona_prompt = await self._persona_alert_context(session)
        try:
            rendered_template = template.format(**fields)
        except (KeyError, ValueError):
            rendered_template = DEFAULT_ALERT_MESSAGE.format(**fields)
        return await self._generate_persona_alert(
            rendered_template, fields, persona_prompt, session
        )

    async def _recovery_text(
        self,
        provider: str,
        balance: Decimal,
        threshold: Decimal,
        session: str | None = None,
    ) -> str:
        fields = self._template_fields(provider, balance, threshold)
        _, persona_prompt = await self._persona_alert_context(session)
        template = self._string_config(
            "recovery_message_template", DEFAULT_RECOVERY_MESSAGE
        )
        try:
            rendered_template = template.format(**fields)
        except (KeyError, ValueError):
            rendered_template = DEFAULT_RECOVERY_MESSAGE.format(**fields)
        return await self._generate_persona_alert(
            rendered_template, fields, persona_prompt, session
        )

    def _template_fields(
        self, provider: str, balance: Decimal, threshold: Decimal
    ) -> dict[str, str]:
        return {
            "account_name": self._provider_display_name(provider),
            "balance": f"{balance:.2f}",
            "threshold": f"{threshold:.2f}",
            "unit": self._provider_currency(provider),
        }

    async def _persona_alert_context(
        self, session: str | None
    ) -> tuple[str, str | None]:
        default_template = self._string_config(
            "alert_message_template", DEFAULT_ALERT_MESSAGE
        )
        if not session:
            return default_template, None

        conversation_manager = getattr(self.context, "conversation_manager", None)
        if conversation_manager is None:
            return default_template, None
        conversation_persona_id = None
        try:
            conversation_id = await conversation_manager.get_curr_conversation_id(
                session
            )
            if not conversation_id:
                conversation = None
            else:
                conversation = await conversation_manager.get_conversation(
                    session, conversation_id
                )
            conversation_persona_id = getattr(conversation, "persona_id", None)
        except Exception:
            logger.exception("Failed to resolve persona for alert session %s.", session)
            return default_template, None

        persona_id = conversation_persona_id
        persona_prompt = None
        persona_manager = getattr(self.context, "persona_manager", None)
        if persona_manager is not None and hasattr(
            persona_manager, "resolve_selected_persona"
        ):
            try:
                resolved_id, persona, _, _ = (
                    await persona_manager.resolve_selected_persona(
                        umo=session,
                        conversation_persona_id=conversation_persona_id,
                        platform_name=session.split(":", 1)[0],
                    )
                )
                persona_id = resolved_id
                if isinstance(persona, Mapping):
                    persona_prompt = persona.get("prompt")
                else:
                    persona_prompt = getattr(persona, "prompt", None)
                    if not persona_prompt:
                        try:
                            persona_prompt = persona["prompt"]
                        except (KeyError, TypeError, IndexError):
                            persona_prompt = None
            except Exception:
                logger.exception("Failed to resolve Persona details for %s.", session)

        if not persona_prompt:
            logger.warning(
                "Persona alert generation skipped for session %s: no Persona prompt resolved.",
                session,
            )
        return default_template, persona_prompt

    async def _generate_persona_alert(
        self,
        rendered_template: str,
        fields: Mapping[str, str],
        persona_prompt: str | None,
        session: str | None,
    ) -> str:
        """Let the active Persona rewrite the template without changing facts."""
        if not session or not persona_prompt:
            return rendered_template
        try:
            conversation_manager = getattr(self.context, "conversation_manager", None)
            conversation_id = None
            conversation = None
            contexts: list[dict[str, Any]] = []
            if conversation_manager is not None:
                conversation_id = await conversation_manager.get_curr_conversation_id(
                    session
                )
                if conversation_id:
                    conversation = await conversation_manager.get_conversation(
                        session, conversation_id
                    )
                if conversation is not None:
                    try:
                        history = json.loads(conversation.history or "[]")
                        if isinstance(history, list):
                            contexts = [
                                item for item in history if isinstance(item, dict)
                            ]
                    except (TypeError, json.JSONDecodeError):
                        logger.warning(
                            "Failed to read conversation history for alert session %s.",
                            session,
                        )
            provider_id = await self.context.get_current_chat_provider_id(session)
            temperature = self._alert_temperature()
            logger.info(
                "Generating Persona-based balance message in native conversation "
                "for session %s with provider %s.",
                session,
                provider_id,
            )
            prompt = (
                "用当前 Persona 的语气，把下面这条余额提醒改写成一条可以直接发给用户的消息。\n"
                "保留这些事实值，不要改动、四舍五入或翻译："
                f"账户={fields['account_name']}，余额={fields['balance']}，"
                f"阈值={fields['threshold']}，单位={fields['unit']}。\n\n"
                f"余额提醒模板：\n{rendered_template}"
            )
            constrained_persona_prompt = f"{persona_prompt}\n\n用你的人格口吻直接说出这条消息，不要解释。"
            response = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=constrained_persona_prompt,
                contexts=contexts,
                temperature=temperature,
            )
            generated = self._clean_generated_message(response.completion_text or "")
            if generated and all(value in generated for value in fields.values()):
                if conversation_manager is not None:
                    if conversation_id is None:
                        conversation_id = await conversation_manager.new_conversation(
                            session, session.split(":", 1)[0]
                        )
                    await conversation_manager.add_message_pair(
                        conversation_id,
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": generated},
                    )
                logger.info(
                    "Persona-based balance message generated and saved to native "
                    "conversation for session %s.",
                    session,
                )
                return generated
            logger.warning(
                "Persona-generated balance message did not preserve all balance facts; using template."
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to generate Persona-based balance message.")
        return rendered_template

    @staticmethod
    def _clean_generated_message(generated: str) -> str:
        generated = generated.strip()
        if generated.startswith("```"):
            generated = re.sub(r"^```[a-zA-Z]*\s*\n?", "", generated, count=1)
            generated = re.sub(r"\n?```\s*$", "", generated, count=1).strip()
        generated = generated.strip('"\'“”‘’「」『』`').strip()
        generated = re.sub(
            r"^(好的|好嘞|没问题|当然|以下是|这是|改写后|改写结果)[，,：:！!。\s]*",
            "",
            generated,
        ).strip()
        generated = re.sub(
            r"[（(]\s*说明[:：].*?[)）]\s*$", "", generated, flags=re.DOTALL
        ).strip()
        return generated

    def _alert_temperature(self) -> float:
        """Return a clamped temperature; invalid config falls back to 1.0."""
        try:
            value = Decimal(str(self.config.get("alert_temperature", "1.0")))
        except (InvalidOperation, ValueError, TypeError):
            logger.warning("配置项 alert_temperature 不是有效数值，将使用默认值 1.0。")
            return 1.0
        if not value.is_finite():
            logger.warning("配置项 alert_temperature 不是有限数值，将使用默认值 1.0。")
            return 1.0
        return float(min(Decimal("2"), max(Decimal("0"), value)))

    def _provider_status_text(
        self, provider: str, balance: Decimal, threshold: Decimal
    ) -> str:
        unit = self._provider_currency(provider)
        level = "余额不足" if balance < threshold else "余额充足"
        return (
            f"{self._provider_display_name(provider)} 当前余额："
            f"{balance:.2f} {unit}\n"
            f"告警阈值：{threshold:.2f} {unit}\n状态：{level}"
        )

    @staticmethod
    def _provider_order() -> tuple[str, ...]:
        return ("deepseek", "derouter", "custom")

    @staticmethod
    def _provider_label(provider: str) -> str:
        if provider == "deepseek":
            return "DeepSeek"
        if provider == "derouter":
            return "derouter"
        return "自定义接口"

    def _enabled_providers(self) -> list[str]:
        return [
            provider
            for provider in self._provider_order()
            if self._bool_config(f"{provider}_enabled")
        ]

    def _preview_providers(self) -> list[str]:
        """Providers used by test commands: enabled ones, or all when none is on."""
        return self._enabled_providers() or list(self._provider_order())

    def _no_provider_text(self) -> str:
        return (
            "尚未启用任何余额监控平台，请先在插件配置中打开对应开关："
            "启用 DeepSeek 余额监控、启用 derouter 余额监控、启用自定义接口余额监控。"
        )

    def _provider_display_name(self, provider: str) -> str:
        if provider == "deepseek":
            return (
                self._string_config("deepseek_account_name", "DeepSeek").strip()
                or "DeepSeek"
            )
        if provider == "derouter":
            return (
                self._string_config("derouter_account_name", "derouter").strip()
                or "derouter"
            )
        return (
            self._string_config("custom_account_name", "自定义 AI API").strip()
            or "自定义 AI API"
        )

    def _provider_currency(self, provider: str) -> str:
        if provider == "deepseek":
            return DEEPSEEK_CURRENCY
        if provider == "derouter":
            return (
                self._string_config("derouter_balance_unit", DEROUTER_CURRENCY).strip()
                or DEROUTER_CURRENCY
            )
        return self._string_config("custom_balance_unit", "USD").strip() or "USD"

    def _derouter_api_key(self) -> str:
        api_key = self._string_config("derouter_api_key").strip()
        if api_key.lower().startswith("bearer "):
            api_key = api_key[7:].strip()
        if not api_key:
            raise BalanceQueryError("请先在插件配置中填写 derouter 密钥")
        if "\r" in api_key or "\n" in api_key:
            raise BalanceQueryError("derouter 密钥格式无效")
        return api_key

    def _derouter_key_type(self) -> str:
        value = self._string_config("derouter_key_type", "账户密钥")
        if value.strip().lower() in {"客户密钥", "customer", "sub-key", "subkey"}:
            return "customer"
        return "account"

    def _bool_config(self, key: str, default: bool = False) -> bool:
        return bool(self.config.get(key, default))

    def _migrate_legacy_config(self) -> None:
        """Move the previous single-platform config onto the flat per-provider keys."""
        legacy_platform = self._string_config("platform").strip().lower()
        if not legacy_platform:
            return

        legacy_derouter = self.config.get("derouter_api")
        legacy_derouter = legacy_derouter if isinstance(legacy_derouter, Mapping) else {}
        legacy_custom = self.config.get("custom_api")
        legacy_custom = legacy_custom if isinstance(legacy_custom, Mapping) else {}

        def legacy_text(source: Mapping[str, Any], key: str) -> str:
            value = source.get(key)
            return value.strip() if isinstance(value, str) else ""

        updates: dict[str, Any] = {}
        if legacy_platform == "deepseek":
            legacy_key = self._string_config("api_key").strip()
            if legacy_key:
                updates["deepseek_api_key"] = legacy_key
                updates["deepseek_enabled"] = True
        elif legacy_platform in {"derouter", "de_router", "derouter.ai"}:
            for new_key, old_key in (
                ("derouter_api_key", "api_key"),
                ("derouter_key_type", "key_type"),
                ("derouter_base_url", "base_url"),
                ("derouter_account_name", "account_name"),
                ("derouter_balance_unit", "balance_unit"),
            ):
                value = legacy_text(legacy_derouter, old_key)
                if value:
                    updates[new_key] = value
            if str(updates.get("derouter_api_key", "")).strip():
                updates["derouter_enabled"] = True
        elif legacy_platform in {"custom", "自定义接口"}:
            for new_key, old_key in (
                ("custom_account_name", "account_name"),
                ("custom_balance_api_url", "balance_api_url"),
                ("custom_balance_api_method", "balance_api_method"),
                ("custom_api_token", "api_token"),
                ("custom_token_header_name", "token_header_name"),
                ("custom_headers_json", "custom_headers_json"),
                ("custom_balance_api_body_json", "balance_api_body_json"),
                ("custom_balance_json_path", "balance_json_path"),
                ("custom_balance_unit", "balance_unit"),
            ):
                value = legacy_text(legacy_custom, old_key)
                if value:
                    updates[new_key] = value
            legacy_scale = legacy_custom.get("balance_scale")
            if (
                isinstance(legacy_scale, (int, float))
                and not isinstance(legacy_scale, bool)
                and legacy_scale > 0
            ):
                updates["custom_balance_scale"] = float(legacy_scale)
            if str(updates.get("custom_balance_api_url", "")).strip():
                updates["custom_enabled"] = True

        # 清理旧字段，保证迁移只执行一次
        updates["platform"] = ""
        updates["api_key"] = ""
        updates["derouter_api"] = {}
        updates["custom_api"] = {}

        for key, value in updates.items():
            self.config[key] = value
        self.config.save_config()
        logger.info(
            "INeedMoney migrated legacy single-platform config (%s) to the "
            "flat multi-provider layout.",
            legacy_platform,
        )

    @staticmethod
    def _to_balance(value: Any, source: str) -> Decimal:
        if value is None or isinstance(value, bool):
            raise BalanceQueryError(f"{source}不是数值")
        try:
            balance = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise BalanceQueryError(f"{source}不是可识别的数值") from exc
        if not balance.is_finite():
            raise BalanceQueryError(f"{source}不是有限数值")
        return balance

    def _alert_is_due(self, provider: str) -> bool:
        last_alert_at = self._last_alert_at.get(provider)
        if last_alert_at is None:
            return True
        cooldown = max(0, self._int_config("alert_cooldown_minutes", 360)) * 60
        return time.monotonic() - last_alert_at >= cooldown

    def _receipt_image_sources(self) -> list[str]:
        """Resolve every configured receipt code, keeping the upload order."""
        value = self.config.get("receipt_code_image", [])
        entries = value if isinstance(value, list) else [value]
        sources: list[str] = []
        for entry in entries:
            source = self._resolve_receipt_image(entry)
            if source and source not in sources:
                sources.append(source)
        return sources

    def _resolve_receipt_image(self, entry: Any) -> str:
        """Handle string, url and object variants returned by AstrBot file fields."""
        if isinstance(entry, Mapping):
            entry = next(
                (
                    entry.get(key)
                    for key in ("path", "file", "url", "value")
                    if entry.get(key)
                ),
                "",
            )
        if not isinstance(entry, str):
            return ""

        image = entry.strip()
        if not image:
            return ""
        if image.startswith(("http://", "https://")):
            return image

        candidates = [Path(image)]
        normalized = image.replace("\\", "/").lstrip("/")
        if normalized.startswith("AstrBot/"):
            normalized = normalized.removeprefix("AstrBot/")
        if normalized.startswith("files/"):
            candidates.append(
                Path(get_astrbot_plugin_data_path()) / "ineedmoney" / normalized
            )

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        logger.warning("Receipt code image does not exist and was skipped: %s", image)
        return ""

    def _append_receipt_images(self, chain: MessageChain) -> None:
        for image in self._receipt_image_sources():
            if image.startswith(("http://", "https://")):
                chain.url_image(image)
            else:
                chain.file_image(image)

    def _enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    def _poll_interval_seconds(self) -> int:
        return max(60, self._int_config("poll_interval_minutes", 10) * 60)

    def _string_config(self, key: str, default: str = "") -> str:
        value = self.config.get(key, default)
        return value if isinstance(value, str) else default

    def _string_list_config(self, key: str) -> list[str]:
        value = self.config.get(key, [])
        if not isinstance(value, list):
            return []
        return list(
            dict.fromkeys(
                item.strip() for item in value if isinstance(item, str) and item.strip()
            )
        )

    def _int_config(self, key: str, default: int) -> int:
        try:
            return int(self.config.get(key, default))
        except (TypeError, ValueError):
            return default

    def _decimal_config(self, key: str, default: str = "0") -> Decimal:
        try:
            value = Decimal(str(self.config.get(key, default)))
        except (InvalidOperation, ValueError, TypeError):
            raise BalanceQueryError(f"配置项 {key} 不是有效数值") from None
        if not value.is_finite():
            raise BalanceQueryError(f"配置项 {key} 不是有限数值")
        return value
