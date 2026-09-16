"""AI API balance monitoring plugin for AstrBot."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlparse

import aiohttp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register


DEFAULT_ALERT_MESSAGE = (
    "[余额告警]\n{account_name} 当前余额为 {balance} {unit}，"
    "已低于告警阈值 {threshold} {unit}。\n请及时充值，避免服务中断。"
)


class BalanceQueryError(RuntimeError):
    """A user-actionable error while reading the configured balance API."""


@register(
    "ineedmoney",
    "rog",
    "定时检查 AI API 余额，并在余额不足时主动发送充值提醒。",
    "1.0.0",
)
class INeedMoneyPlugin(Star):
    """Query a configurable JSON endpoint and proactively notify allowlisted chats."""

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
        url = self._string_config("balance_api_url").strip()
        if not url:
            raise BalanceQueryError("尚未配置余额 API 地址")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise BalanceQueryError("余额 API 地址必须是有效的 http 或 https URL")

        method = self._string_config("balance_api_method", "GET").upper()
        if method not in {"GET", "POST"}:
            raise BalanceQueryError("余额 API 请求方法仅支持 GET 或 POST")
        request_kwargs: dict[str, Any] = {"headers": self._build_headers()}
        if method == "POST":
            request_kwargs["json"] = self._request_json_body()

        timeout = aiohttp.ClientTimeout(
            total=max(1, self._int_config("request_timeout_seconds", 15))
        )
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(method, url, **request_kwargs) as response:
                    response_text = await response.text()
                    if not 200 <= response.status < 300:
                        raise BalanceQueryError(f"余额 API 返回 HTTP {response.status}")
        except asyncio.CancelledError:
            raise
        except BalanceQueryError:
            raise
        except asyncio.TimeoutError as exc:
            raise BalanceQueryError("余额 API 请求超时") from exc
        except aiohttp.ClientError as exc:
            raise BalanceQueryError(f"无法连接余额 API：{exc}") from exc

        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise BalanceQueryError("余额 API 未返回 JSON 数据") from exc
        value = self._extract_json_path(
            payload, self._string_config("balance_json_path", "data.balance")
        )
        if value is None or isinstance(value, bool):
            raise BalanceQueryError("余额字段不是数值")
        try:
            balance = Decimal(str(value)) * self._decimal_config("balance_scale", "1")
        except (InvalidOperation, ValueError) as exc:
            raise BalanceQueryError("余额字段不是可识别的数值") from exc
        if not balance.is_finite():
            raise BalanceQueryError("余额字段不是有限数值")
        return balance

    def _build_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        token = self._string_config("balance_api_token")
        if token:
            name = self._string_config("token_header_name", "Authorization")
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

    def _request_json_body(self) -> Any:
        body = self._string_config("balance_api_body_json")
        if not body.strip():
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise BalanceQueryError("请求体不是合法 JSON") from exc

    @staticmethod
    def _extract_json_path(payload: Any, path: str) -> Any:
        """Read dotted object keys and numeric indexes, e.g. data.users[0].balance."""
        path = path.strip().removeprefix("$").lstrip(".")
        if not path:
            raise BalanceQueryError("余额 JSON 路径不能为空")
        current = payload
        for segment in path.split("."):
            match = re.fullmatch(r"([^\[\]]+)((?:\[\d+\])*)", segment)
            if not match:
                raise BalanceQueryError(f"余额 JSON 路径格式无效：{path}")
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
            chain = MessageChain().message(self._alert_text(balance, threshold))
            if image:
                if image.startswith(("http://", "https://")):
                    chain.url_image(image)
                else:
                    chain.file_image(image)
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

    def _alert_text(self, balance: Decimal, threshold: Decimal) -> str:
        precision = min(8, max(0, self._int_config("display_precision", 2)))
        fields = {
            "account_name": self._string_config("account_name", "AI API"),
            "balance": f"{balance:.{precision}f}",
            "threshold": f"{threshold:.{precision}f}",
            "unit": self._string_config("balance_unit", "USD"),
        }
        template = self._string_config("alert_message", DEFAULT_ALERT_MESSAGE)
        try:
            return template.format(**fields)
        except (KeyError, ValueError, IndexError):
            logger.warning("Invalid alert message template; using the default template.")
            return DEFAULT_ALERT_MESSAGE.format(**fields)

    def _balance_status_text(self, balance: Decimal) -> str:
        threshold = self._decimal_config("low_balance_threshold")
        precision = min(8, max(0, self._int_config("display_precision", 2)))
        unit = self._string_config("balance_unit", "USD")
        level = "余额不足" if balance < threshold else "余额充足"
        return (
            f"{self._string_config('account_name', 'AI API')} 当前余额："
            f"{balance:.{precision}f} {unit}\n"
            f"告警阈值：{threshold:.{precision}f} {unit}\n状态：{level}"
        )

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
        return value.strip() if isinstance(value, str) else ""

    def _enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    def _poll_interval_seconds(self) -> int:
        return max(30, self._int_config("poll_interval_minutes", 10) * 60)

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
