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
DEEPSEEK_BALANCE_URL = "https://api.deepseek.com/user/balance"
DEEPSEEK_CURRENCY = "CNY"
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
        self._last_alert_at: float | None = None
        self._last_balance: Decimal | None = None
        self._last_query_error: str | None = None
        self._is_low_balance = False

    async def initialize(self) -> None:
        """Start one lifecycle-managed task; it rereads WebUI settings each cycle."""
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
        """管理员：立即查询余额并验证当前接口配置。"""
        if not event.is_admin():
            yield event.plain_result("此命令仅 AstrBot 管理员可用。")
            return
        try:
            balance = await self._query_balance()
            self._last_balance, self._last_query_error = balance, None
            yield event.plain_result(self._balance_status_text(balance))
        except BalanceQueryError as exc:
            yield event.plain_result(f"余额查询失败：{exc}")

    @filter.command("余额告警测试")
    async def test_alert_in_chat(self, event: AstrMessageEvent):
        """管理员：使用示例低余额在当前会话测试告警样式。"""
        if not event.is_admin():
            yield event.plain_result("此命令仅 AstrBot 管理员可用。")
            return
        try:
            threshold = self._decimal_config("low_balance_threshold")
            if threshold < 0:
                raise BalanceQueryError("配置项 low_balance_threshold 不能小于 0")
            test_threshold = threshold if threshold > 0 else Decimal("1")
            test_balance = test_threshold / Decimal("2")
            chain = MessageChain().message(
                "[告警样式测试]\n"
                + await self._alert_text(
                    test_balance, test_threshold, event.unified_msg_origin
                )
            )
            image = self._receipt_image_source()
            if image:
                if image.startswith(("http://", "https://")):
                    chain.url_image(image)
                elif Path(image).is_file():
                    chain.file_image(image)
                else:
                    logger.warning(
                        "Receipt code image does not exist, sending alert without it: %s",
                        image,
                    )
            yield event.chain_result(chain)
        except BalanceQueryError as exc:
            yield event.plain_result(f"余额告警测试失败：{exc}")

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
        """管理员：查看后台最近一次查询，不额外发起 API 请求。"""
        if not event.is_admin():
            yield event.plain_result("此命令仅 AstrBot 管理员可用。")
            return
        if self._last_query_error:
            yield event.plain_result(f"余额监控最近一次查询失败：{self._last_query_error}")
        elif self._last_balance is None:
            yield event.plain_result("余额监控尚未完成一次查询。")
        else:
            yield event.plain_result(self._balance_status_text(self._last_balance))

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
            try:
                balance = await self._query_balance()
                self._last_balance, self._last_query_error = balance, None
                threshold = self._decimal_config("low_balance_threshold")
                if threshold < 0:
                    raise BalanceQueryError("配置项 low_balance_threshold 不能小于 0")
            except BalanceQueryError as exc:
                self._last_query_error = str(exc)
                logger.warning("Balance query failed: %s", exc)
                return

            if balance >= threshold:
                if self._is_low_balance:
                    logger.info("Balance recovered above the configured threshold.")
                self._is_low_balance, self._last_alert_at = False, None
                return

            self._is_low_balance = True
            if not self._alert_is_due():
                return
            if await self._send_low_balance_alert(balance, threshold):
                self._last_alert_at = time.monotonic()

    async def _query_balance(self) -> Decimal:
        platform = self._platform_key()
        if platform == "deepseek":
            return await self._query_deepseek_balance()
        if platform == "custom":
            return await self._query_custom_balance()
        raise BalanceQueryError("暂不支持所选余额查询平台")

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
        """Return a bare DeepSeek key; accept the old Bearer form during migration."""
        api_key = self._string_config("api_key").strip()
        if not api_key:
            api_key = self._string_config("balance_api_token").strip()
        if api_key.lower().startswith("bearer "):
            api_key = api_key[7:].strip()
        if not api_key:
            raise BalanceQueryError("请先在插件配置中填写 DeepSeek API Key")
        if "\r" in api_key or "\n" in api_key:
            raise BalanceQueryError("DeepSeek API Key 格式无效")
        return api_key

    async def _query_custom_balance(self) -> Decimal:
        """Query an advanced, user-supplied JSON balance endpoint."""
        custom = self._custom_api_config()
        url = self._custom_string_config(custom, "balance_api_url").strip()
        if not url:
            raise BalanceQueryError("请先在自定义接口配置中填写余额 API 地址")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise BalanceQueryError("自定义余额 API 地址必须是有效的 http 或 https URL")

        method = self._custom_string_config(custom, "balance_api_method", "GET").upper()
        if method not in {"GET", "POST"}:
            raise BalanceQueryError("自定义余额 API 请求方法仅支持 GET 或 POST")
        request_kwargs: dict[str, Any] = {
            "headers": self._build_custom_headers(custom)
        }
        if method == "POST":
            request_kwargs["json"] = self._custom_request_json_body(custom)

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
            payload,
            self._custom_string_config(custom, "balance_json_path", "data.balance"),
        )
        if value is None or isinstance(value, bool):
            raise BalanceQueryError("自定义接口返回的余额不是数值")
        try:
            balance = Decimal(str(value)) * self._custom_decimal_config(
                custom, "balance_scale", "1"
            )
        except (InvalidOperation, ValueError) as exc:
            raise BalanceQueryError("自定义接口返回的余额不是可识别的数值") from exc
        if not balance.is_finite():
            raise BalanceQueryError("自定义接口返回的余额不是有限数值")
        return balance

    def _build_custom_headers(self, custom: Mapping[str, Any]) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        token = self._custom_string_config(custom, "api_token")
        if token:
            name = self._custom_string_config(
                custom, "token_header_name", "Authorization"
            )
            self._validate_header(name, token)
            headers[name] = token

        custom_headers_text = self._custom_string_config(custom, "custom_headers_json")
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

    def _custom_request_json_body(self, custom: Mapping[str, Any]) -> Any:
        body = self._custom_string_config(custom, "balance_api_body_json")
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
        self, balance: Decimal, threshold: Decimal
    ) -> bool:
        sessions = self._string_list_config("notification_sessions")
        if not sessions:
            logger.warning("Balance is low, but no notification sessions are configured.")
            return False
        delivered, image = False, self._receipt_image_source()
        for session in sessions:
            chain = MessageChain().message(
                await self._alert_text(balance, threshold, session)
            )
            if image:
                if image.startswith(("http://", "https://")):
                    chain.url_image(image)
                elif Path(image).is_file():
                    chain.file_image(image)
                else:
                    logger.warning(
                        "Receipt code image does not exist, sending alert without it: %s",
                        image,
                    )
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
        self, balance: Decimal, threshold: Decimal, session: str | None = None
    ) -> str:
        fields = {
            "account_name": self._platform_display_name(),
            "balance": f"{balance:.2f}",
            "threshold": f"{threshold:.2f}",
            "unit": self._platform_currency(),
        }
        template, persona_prompt = await self._persona_alert_context(session)
        try:
            rendered_template = template.format(**fields)
        except (KeyError, ValueError):
            rendered_template = DEFAULT_ALERT_MESSAGE.format(**fields)
        return await self._generate_persona_alert(
            rendered_template, fields, persona_prompt, session
        )

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
            except Exception:
                logger.exception("Failed to resolve Persona details for %s.", session)

        if not persona_id:
            return default_template, persona_prompt
        templates = self.config.get("alert_persona_templates", [])
        if not isinstance(templates, list):
            return default_template, persona_prompt
        for item in templates:
            if not isinstance(item, Mapping):
                continue
            if item.get("persona_id") == persona_id:
                template = item.get("template")
                if isinstance(template, str) and template:
                    return template, persona_prompt
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
            provider_id = await self.context.get_current_chat_provider_id(session)
            prompt = (
                "请把下面的余额提醒模板改写成一条可以直接发送给用户的消息。\n"
                "必须遵循当前 Persona 的语气，但不要解释改写过程，不要添加标题、Markdown 或引号。\n"
                "必须保留以下事实值，不能修改、四舍五入、翻译或省略："
                f"账户={fields['account_name']}，余额={fields['balance']}，"
                f"阈值={fields['threshold']}，单位={fields['unit']}。\n\n"
                f"余额提醒模板：\n{rendered_template}"
            )
            response = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=persona_prompt,
            )
            generated = (response.completion_text or "").strip()
            if generated and all(value in generated for value in fields.values()):
                return generated
            logger.warning(
                "Persona-generated alert did not preserve all balance facts; using template."
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to generate Persona-based balance alert.")
        return rendered_template

    def _balance_status_text(self, balance: Decimal) -> str:
        threshold = self._decimal_config("low_balance_threshold")
        level = "余额不足" if balance < threshold else "余额充足"
        return (
            f"{self._platform_display_name()} 当前余额："
            f"{balance:.2f} {self._platform_currency()}\n"
            f"告警阈值：{threshold:.2f} {self._platform_currency()}\n状态：{level}"
        )

    def _platform_display_name(self) -> str:
        if self._platform_key() == "deepseek":
            return "DeepSeek"
        return self._custom_string_config(
            self._custom_api_config(), "account_name", "自定义 AI API"
        )

    def _platform_currency(self) -> str:
        if self._platform_key() == "deepseek":
            return DEEPSEEK_CURRENCY
        return self._custom_string_config(
            self._custom_api_config(), "balance_unit", "USD"
        )

    def _platform_key(self) -> str:
        platform = self._string_config("platform", "deepseek").strip().lower()
        return "custom" if platform in {"custom", "自定义接口"} else platform

    def _custom_api_config(self) -> Mapping[str, Any]:
        value = self.config.get("custom_api", {})
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _custom_string_config(
        custom: Mapping[str, Any], key: str, default: str = ""
    ) -> str:
        value = custom.get(key, default)
        return value if isinstance(value, str) else default

    @staticmethod
    def _custom_decimal_config(
        custom: Mapping[str, Any], key: str, default: str = "0"
    ) -> Decimal:
        try:
            value = Decimal(str(custom.get(key, default)))
        except (InvalidOperation, ValueError, TypeError):
            raise BalanceQueryError(f"自定义接口配置项 {key} 不是有效数值") from None
        if not value.is_finite():
            raise BalanceQueryError(f"自定义接口配置项 {key} 不是有限数值")
        return value

    def _alert_is_due(self) -> bool:
        if self._last_alert_at is None:
            return True
        cooldown = max(0, self._int_config("alert_cooldown_minutes", 360)) * 60
        return time.monotonic() - self._last_alert_at >= cooldown

    def _receipt_image_source(self) -> str:
        """Handle both string and object variants returned by AstrBot file fields."""
        value = self.config.get("receipt_code_image", [])
        if isinstance(value, list):
            value = value[0] if value else ""
        if isinstance(value, Mapping):
            value = next(
                (value.get(key) for key in ("path", "file", "url", "value") if value.get(key)),
                "",
            )
        if not isinstance(value, str):
            return ""

        image = value.strip()
        if not image or image.startswith(("http://", "https://")):
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
        return image

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
