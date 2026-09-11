"""Exercise ctypes ownership without loading a library or touching USB."""
import ctypes
import gc
import weakref

import pytest

from stdatalog_core.HSD_link.communication.PnPL_HSD import hsd_dll as sdk


class NativeStub:
    """The shipped C++ callback map is insert-only and stores raw pointers."""

    def __init__(self):
        self.callbacks = {}
        self.registrations = 0
        self.registration_result = 0
        self.close_result = 0
        self.command_result = 0
        self.response = ctypes.create_string_buffer(b'{"status":true}')
        self.freed = []

    def hs_datalog_set_data_ready_callback(self, device, name, callback):
        self.registrations += 1
        if self.registration_result == 0:
            self.callbacks.setdefault((device.value, name.value),
                                      ctypes.cast(callback, ctypes.c_void_p).value)
        return self.registration_result

    def hs_datalog_close(self):
        if self.close_result == 0:
            self.callbacks.clear()
        return self.close_result

    def command(self, *args):
        # Even an error can expose a non-NULL, unowned/dangling response pointer.
        ctypes.cast(args[-1], ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.addressof(self.response)
        return self.command_result

    hs_datalog_start_log = command
    hs_datalog_stop_log = command
    hs_datalog_set_rtc_time = command

    def hs_datalog_free(self, ptr):
        self.freed.append(ctypes.cast(ptr, ctypes.c_void_p).value)
        return 0


@pytest.fixture
def dll(monkeypatch):
    native = NativeStub()
    monkeypatch.setattr(sdk, "HSD_Dll_Wrapper", lambda: native)
    return sdk.HSD_Dll(), native


@pytest.mark.parametrize("method,args", [
    ("hs_datalog_start_log", (0, 1)),
    ("hs_datalog_stop_log", (0,)),
    ("hs_datalog_set_rtc_time", (0,)),
])
@pytest.mark.parametrize("result", [-1, -2, 3])
def test_failed_command_never_reads_or_frees_unowned_response(dll, method, args, result):
    link, native = dll
    native.command_result = result
    native.response = ctypes.create_string_buffer(b"\xff")
    assert getattr(link, method)(*args)[0] is False
    assert native.freed == []


@pytest.mark.parametrize("response", [b'{"status":true}', b"", b"\xff"])
def test_successful_response_freed_once_even_if_decode_fails(dll, response):
    link, native = dll
    native.response = ctypes.create_string_buffer(response)
    if response == b"\xff":
        with pytest.raises(UnicodeDecodeError):
            link.hs_datalog_start_log(0, 1)
    else:
        assert link.hs_datalog_start_log(0, 1)[0] is True
    assert native.freed == [ctypes.addressof(native.response)]


def test_stable_callback_ignores_dangling_native_name_and_supports_replace_disable(dll):
    link, native = dll
    calls = []
    first = lambda *args: calls.append(("first", args)) or 0
    second = lambda *args: calls.append(("second", args)) or 0
    key = (2, "iis3dwb_acc")
    assert link.hs_datalog_set_data_ready_callback(*key, first)
    address = native.callbacks[(2, b"iis3dwb_acc")]
    thunk_ref = weakref.ref(link._data_ready_callbacks[key])
    assert link.hs_datalog_set_data_ready_callback(*key, second)
    gc.collect()
    assert thunk_ref() is not None, "Native still owns the FIRST callback address"
    assert native.registrations == 1
    assert sdk.HSD_DATA_READY_CALLBACK._argtypes_[1] is ctypes.c_void_p
    payload = (ctypes.c_uint8 * 2)(1, 2)
    # Address 1 deliberately cannot be dereferenced. Only the registered name is safe.
    invoke = sdk.HSD_DATA_READY_CALLBACK(address)
    assert invoke(2, 1, payload, 2) == 0
    assert calls[-1][0] == "second"
    assert calls[-1][1][:2] == (2, b"iis3dwb_acc")
    assert link.hs_datalog_set_data_ready_callback(*key, None)
    gc.collect()
    assert thunk_ref() is not None
    assert invoke(2, 1, payload, 2) == 0
    assert len(calls) == 1
    assert link.hs_datalog_set_data_ready_callback(*key, first)
    assert invoke(2, 1, payload, 2) == 0
    assert calls[-1][0] == "first"
    assert invoke(99, 1, payload, 2) == 0
    assert calls[-1][1][0] == 99, "Never replace native device identity"

    def broken_callback(*_args):
        raise RuntimeError("callback failed")

    assert link.hs_datalog_set_data_ready_callback(*key, broken_callback)
    assert invoke(2, 1, payload, 2) == -1
    assert native.registrations == 1
    native.close_result = -1
    assert link.hs_datalog_close() is False
    assert thunk_ref() is not None
    native.close_result = 0
    assert link.hs_datalog_close() is True
    gc.collect()
    assert thunk_ref() is None
    assert not link._data_ready_callbacks


def test_registration_error_can_retry_and_preserves_no_native_pointer(dll):
    link, native = dll
    native.registration_result = -1
    assert not link.hs_datalog_set_data_ready_callback(0, "iis3dwb_acc", lambda *_: 0)
    assert not link._data_ready_callbacks
    assert not native.callbacks
    native.registration_result = 0
    assert link.hs_datalog_set_data_ready_callback(0, "iis3dwb_acc", lambda *_: 0)
    assert len(native.callbacks) == 1


@pytest.mark.parametrize("name", ["", "iis3dwb\x00_acc"])
def test_invalid_component_name_rejected_before_native_call(dll, name):
    link, native = dll
    with pytest.raises(ValueError):
        link.hs_datalog_set_data_ready_callback(0, name, lambda *_: 0)
    assert native.registrations == 0
