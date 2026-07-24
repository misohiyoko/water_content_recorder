#!/usr/bin/env python3
import struct
import time

import numpy as np
import serial
from scipy import signal
from serial.tools import list_ports

VID = 0x0483  # 1155
PID = 0x5740  # 22336

_KNOWN_DEVICES = [
    (0x0483, 0x5740),  # NanoVNA (STM32 USB CDC)
    (0x04B4, 0x0008),  # LiteVNA-64 / NanoVNA V2
]

_LITEVNA_DEVICES = {
    (0x04B4, 0x0008),  # LiteVNA-64 / NanoVNA V2
}


# Get nanovna device automatically
def getport() -> str:
    device_list = list_ports.comports()
    for device in device_list:
        if (device.vid, device.pid) in _KNOWN_DEVICES:
            return device.device
    raise OSError("device not found")


REF_LEVEL = 1 << 9


class NanoVNA:
    def __init__(self, dev=None):
        self.dev = dev or getport()
        self.serial: serial.Serial | None = None
        self._frequencies: np.ndarray | None = None
        self.points = 101

    @property
    def frequencies(self):
        return self._frequencies

    def set_frequencies(self, start=1e6, stop=900e6, points=None):
        if points:
            self.points = points
        self._frequencies = np.linspace(start, stop, self.points)

    def open(self):
        if self.serial is None:
            self.serial = serial.Serial(self.dev)

    def close(self):
        if self.serial:
            self.serial.close()
        self.serial = None

    def send_command(self, cmd):
        self.open()
        assert self.serial is not None
        self.serial.write(cmd.encode())
        self.serial.readline()  # discard empty line

    def set_sweep(self, start, stop):
        if start is not None:
            self.send_command("sweep start %d\r" % start)
        if stop is not None:
            self.send_command("sweep stop %d\r" % stop)

    def set_frequency(self, freq):
        if freq is not None:
            self.send_command("freq %d\r" % freq)

    def set_port(self, port):
        if port is not None:
            self.send_command("port %d\r" % port)

    def set_gain(self, gain):
        if gain is not None:
            self.send_command("gain %d %d\r" % (gain, gain))

    def set_offset(self, offset):
        if offset is not None:
            self.send_command("offset %d\r" % offset)

    def set_strength(self, strength):
        if strength is not None:
            self.send_command("power %d\r" % strength)

    def set_filter(self, filter):
        self.filter = filter

    def fetch_data(self):
        assert self.serial is not None
        result = ""
        line = ""
        while True:
            c = self.serial.read().decode("utf-8")
            if c == chr(13):
                continue  # ignore CR
            line += c
            if c == chr(10):
                result += line
                line = ""
                continue
            if line.endswith("ch>"):
                # stop on prompt
                break
        return result

    def fetch_buffer(self, freq=None, buffer=0):
        self.send_command("dump %d\r" % buffer)
        data = self.fetch_data()
        x = []
        for line in data.split("\n"):
            if line:
                x.extend([int(d, 16) for d in line.strip().split(" ")])
        return np.array(x, dtype=np.int16)

    def fetch_rawwave(self, freq=None):
        if freq:
            self.set_frequency(freq)
            time.sleep(0.05)
        self.send_command("dump 0\r")
        data = self.fetch_data()
        x = []
        for line in data.split("\n"):
            if line:
                x.extend([int(d, 16) for d in line.strip().split(" ")])
        return np.array(x[0::2], dtype=np.int16), np.array(x[1::2], dtype=np.int16)

    def fetch_array(self, sel):
        self.send_command("data %d\r" % sel)
        data = self.fetch_data()
        x = []
        for line in data.split("\n"):
            if line:
                x.extend([float(d) for d in line.strip().split(" ")])
        return np.array(x[0::2]) + np.array(x[1::2]) * 1j

    def fetch_gamma(self, freq=None):
        if freq:
            self.set_frequency(freq)
        self.send_command("gamma\r")
        assert self.serial is not None
        data = self.serial.readline()
        d = data.strip().split(b" ")
        return (int(d[0]) + int(d[1]) * 1.0j) / REF_LEVEL

    def reflect_coeff_from_rawwave(self, freq=None):
        ref, samp = self.fetch_rawwave(freq)
        refh = signal.hilbert(ref)
        # x = np.correlate(refh, samp) / np.correlate(refh, refh)
        # return x[0]
        # return np.sum(refh*samp / np.abs(refh) / REF_LEVEL)
        return np.average(refh * samp / np.abs(refh) / REF_LEVEL)

    reflect_coeff = reflect_coeff_from_rawwave
    gamma = reflect_coeff_from_rawwave
    # gamma = fetch_gamma
    coefficient = reflect_coeff

    def resume(self):
        self.send_command("resume\r")

    def pause(self):
        self.send_command("pause\r")

    def scan_gamma0(self, port=None):
        self.set_port(port)
        return np.vectorize(self.gamma)(self.frequencies)

    def scan_gamma(self, port=None):
        self.set_port(port)
        return np.vectorize(self.fetch_gamma)(self.frequencies)

    def data(self, array=0):
        self.send_command("data %d\r" % array)
        data = self.fetch_data()
        x = []
        for line in data.split("\n"):
            if line:
                d = line.strip().split(" ")
                x.append(float(d[0]) + float(d[1]) * 1.0j)
        return np.array(x)

    def fetch_frequencies(self):
        self.send_command("frequencies\r")
        data = self.fetch_data()
        x = []
        for line in data.split("\n"):
            if line:
                x.append(float(line))
        self._frequencies = np.array(x)

    def send_scan(self, start=1e6, stop=900e6, points=None):
        if points:
            self.send_command("scan %d %d %d\r" % (start, stop, points))
        else:
            self.send_command("scan %d %d\r" % (start, stop))

    def scan(self):
        segment_length = 101
        array0 = []
        array1 = []
        if self._frequencies is None:
            self.fetch_frequencies()
        assert self._frequencies is not None
        freqs = self._frequencies
        while len(freqs) > 0:
            seg_start = freqs[0]
            seg_stop = freqs[segment_length - 1] if len(freqs) >= segment_length else freqs[-1]
            length = min(segment_length, len(freqs))
            # print((seg_start, seg_stop, length))
            self.send_scan(seg_start, seg_stop, length)
            array0.extend(self.data(0))
            array1.extend(self.data(1))
            freqs = freqs[segment_length:]
        self.resume()
        return (array0, array1)


class LiteVNA(NanoVNA):
    """LiteVNA / NanoVNA-V2 — binary USB protocol."""

    # Register addresses
    _REG_SWEEP_START = 0x00
    _REG_SWEEP_STEP = 0x10
    _REG_SWEEP_POINTS = 0x20
    _REG_VALUES_PER_FREQ = 0x22
    _REG_VALUES_FIFO = 0x30
    _REG_AVERAGE = 0x40
    _REG_LOW_FREQ_POWER = 0x41
    _REG_HIGH_FREQ_POWER = 0x42
    _REG_CHANNEL_SELECT = 0x44

    # Opcodes
    _CMD_NOP = 0x00
    _CMD_INDICATE = 0x0D
    _CMD_READ1 = 0x10
    _CMD_READ2 = 0x11
    _CMD_READ4 = 0x12
    _CMD_READFIFO = 0x18
    _CMD_WRITE1 = 0x20
    _CMD_WRITE2 = 0x21
    _CMD_WRITE4 = 0x22
    _CMD_WRITE8 = 0x23

    _FIFO_BYTES = 32  # bytes per FIFO entry

    def open(self):
        if self.serial is None:
            self.serial = serial.Serial(self.dev, timeout=5)

    # ---- low-level register I/O ----

    def _r1(self, addr: int) -> int:
        self.serial.write(bytes([self._CMD_READ1, addr]))
        return struct.unpack("B", self.serial.read(1))[0]

    def _r2(self, addr: int) -> int:
        self.serial.write(bytes([self._CMD_READ2, addr]))
        return struct.unpack("<H", self.serial.read(2))[0]

    def _r4(self, addr: int) -> int:
        self.serial.write(bytes([self._CMD_READ4, addr]))
        return struct.unpack("<I", self.serial.read(4))[0]

    def _r8(self, addr: int) -> int:
        return self._r4(addr) | (self._r4(addr + 4) << 32)

    def _readfifo(self, addr: int, count: int) -> bytes:
        result = b""
        remaining = count
        while remaining > 0:
            n = min(remaining, 255)
            self.serial.write(bytes([self._CMD_READFIFO, addr, n]))
            result += self.serial.read(self._FIFO_BYTES * n)
            remaining -= n
        return result

    def _w1(self, addr: int, value: int):
        self.serial.write(bytes([self._CMD_WRITE1, addr, value & 0xFF]))

    def _w2(self, addr: int, value: int):
        self.serial.write(bytes([self._CMD_WRITE2, addr]) + struct.pack("<H", value))

    def _w4(self, addr: int, value: int):
        self.serial.write(bytes([self._CMD_WRITE4, addr]) + struct.pack("<I", value))

    def _w8(self, addr: int, value: int):
        self.serial.write(bytes([self._CMD_WRITE8, addr]) + struct.pack("<Q", value))

    # ---- sweep control ----

    def set_sweep(self, start, stop):
        step = int((stop - start) / (self.points - 1))
        self.open()
        self._w8(self._REG_SWEEP_START, int(start))
        self._w8(self._REG_SWEEP_STEP, step)
        self._w2(self._REG_SWEEP_POINTS, self.points)
        self._w2(self._REG_VALUES_PER_FREQ, 1)
        self._frequencies = np.linspace(start, stop, self.points)

    def fetch_frequencies(self):
        self.open()
        start = self._r8(self._REG_SWEEP_START)
        step = self._r8(self._REG_SWEEP_STEP)
        points = self._r2(self._REG_SWEEP_POINTS)
        self.points = points
        self._frequencies = np.array([start + i * step for i in range(points)])

    # ---- FIFO parsing ----

    def _parse_fifo(self, raw: bytes, points: int) -> list:
        """Parse raw FIFO bytes; return list of (S11, S21) indexed by freqIndex."""
        entries = [(0j, 0j)] * points
        for i in range(points):
            chunk = raw[i * self._FIFO_BYTES : (i + 1) * self._FIFO_BYTES]
            fwd_re, fwd_im, r0_re, r0_im, r1_re, r1_im, idx = struct.unpack("<iiiiiiH6x", chunk)
            fwd = complex(fwd_re, fwd_im)
            if fwd == 0:
                entries[idx] = (0j, 0j)
            else:
                entries[idx] = (complex(r0_re, r0_im) / fwd, complex(r1_re, r1_im) / fwd)
        return entries

    # ---- data acquisition ----

    def data(self, array=0):
        self.open()
        self._w1(self._REG_CHANNEL_SELECT, 0x01 if array == 0 else 0x02)
        points = len(self._frequencies) if self._frequencies is not None else self.points
        self._w1(self._REG_VALUES_FIFO, 0)  # clear stale FIFO data
        raw = self._readfifo(self._REG_VALUES_FIFO, points)
        entries = self._parse_fifo(raw, points)
        return np.array([e[0 if array == 0 else 1] for e in entries])

    def scan(self):
        if self._frequencies is None:
            self.fetch_frequencies()
        points = len(self._frequencies)
        self.open()
        self._w1(self._REG_CHANNEL_SELECT, 0x00)  # both S11 and S21
        self._w1(self._REG_VALUES_FIFO, 0)  # clear stale FIFO data
        raw = self._readfifo(self._REG_VALUES_FIFO, points)
        entries = self._parse_fifo(raw, points)
        return ([e[0] for e in entries], [e[1] for e in entries])

    # ---- stubs for text-protocol methods ----

    def send_command(self, cmd):
        raise NotImplementedError("LiteVNA uses binary protocol, not text commands")

    def resume(self):
        pass

    def pause(self):
        pass


def open_device(dev=None) -> NanoVNA:
    """Return a NanoVNA or LiteVNA instance depending on the connected hardware."""
    if dev is None:
        device_list = list_ports.comports()
        for device in device_list:
            if (device.vid, device.pid) in _LITEVNA_DEVICES:
                return LiteVNA(device.device)
            if (device.vid, device.pid) in set(_KNOWN_DEVICES):
                return NanoVNA(device.device)
        raise OSError("device not found")
    return NanoVNA(dev)
