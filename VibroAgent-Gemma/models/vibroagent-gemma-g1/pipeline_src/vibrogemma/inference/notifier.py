from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass

from ..contracts import ClassificationResult


@dataclass(slots=True)
class Notification:
    title: str
    body: str
    urgency: str


class DesktopNotifier:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def build(self, result: ClassificationResult) -> Notification:
        affected = [target for target in result.targets if target.affected or target.class_name == "data_invalid"]
        if affected:
            sensors = ", ".join(target.sensor_id for target in affected)
            body = f"Affected: {sensors}. {result.explanation}".strip()
        else:
            body = result.explanation or "All targets were classified as normal."
        urgency = "critical" if result.system_state == "critical" else "normal"
        return Notification(f"VibroGemma: {result.system_state}", body[:500], urgency)

    def send(self, result: ClassificationResult) -> bool:
        if not self.enabled:
            return False
        notification = self.build(result)
        system = platform.system().lower()
        try:
            if system == "linux" and shutil.which("notify-send"):
                subprocess.Popen(
                    ["notify-send", "--urgency", notification.urgency, notification.title, notification.body],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            if system == "darwin" and shutil.which("osascript"):
                script = (
                    f'display notification {notification.body!r} with title {notification.title!r}'
                )
                subprocess.Popen(["osascript", "-e", script])
                return True
            if system == "windows" and shutil.which("powershell"):
                escaped_title = notification.title.replace("'", "''")
                escaped_body = notification.body.replace("'", "''")
                command = (
                    "Add-Type -AssemblyName PresentationFramework; "
                    f"[System.Windows.MessageBox]::Show('{escaped_body}','{escaped_title}')"
                )
                subprocess.Popen(["powershell", "-NoProfile", "-Command", command])
                return True
            return self._tkinter(notification)
        except Exception:
            return False

    @staticmethod
    def _tkinter(notification: Notification) -> bool:
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showwarning(notification.title, notification.body)
            root.destroy()
            return True
        except Exception:
            return False
