#!/bin/bash
echo "=== hwmon4 (1-0040) ==="
for f in /sys/devices/platform/c240000.i2c/i2c-1/1-0040/hwmon/hwmon4/*_label; do
  echo -n "$f: "; cat "$f" 2>/dev/null || echo "(no permission)"
done
echo ""
for f in /sys/devices/platform/c240000.i2c/i2c-1/1-0040/hwmon/hwmon4/in*_input; do
  echo -n "$f: "; cat "$f" 2>/dev/null || echo "(no permission)"
done
for f in /sys/devices/platform/c240000.i2c/i2c-1/1-0040/hwmon/hwmon4/curr*_input; do
  echo -n "$f: "; cat "$f" 2>/dev/null || echo "(no permission)"
done

echo "=== hwmon5 (1-0041) ==="
for f in /sys/devices/platform/c240000.i2c/i2c-1/1-0041/hwmon/hwmon5/*_label; do
  echo -n "$f: "; cat "$f" 2>/dev/null || echo "(no permission)"
done
echo ""
for f in /sys/devices/platform/c240000.i2c/i2c-1/1-0041/hwmon/hwmon5/in*_input; do
  echo -n "$f: "; cat "$f" 2>/dev/null || echo "(no permission)"
done
for f in /sys/devices/platform/c240000.i2c/i2c-1/1-0041/hwmon/hwmon5/curr*_input; do
  echo -n "$f: "; cat "$f" 2>/dev/null || echo "(no permission)"
done
