#!/bin/bash
# Run with sudo to fix hwmon permissions
chmod a+r /sys/devices/platform/c240000.i2c/i2c-1/1-0040/hwmon/hwmon4/in1_input
chmod a+r /sys/devices/platform/c240000.i2c/i2c-1/1-0040/hwmon/hwmon4/in2_input
chmod a+r /sys/devices/platform/c240000.i2c/i2c-1/1-0040/hwmon/hwmon4/in3_input
chmod a+r /sys/devices/platform/c240000.i2c/i2c-1/1-0040/hwmon/hwmon4/curr1_input
chmod a+r /sys/devices/platform/c240000.i2c/i2c-1/1-0040/hwmon/hwmon4/curr2_input
chmod a+r /sys/devices/platform/c240000.i2c/i2c-1/1-0040/hwmon/hwmon4/curr3_input
chmod a+r /sys/devices/platform/c240000.i2c/i2c-1/1-0041/hwmon/hwmon5/in1_input
chmod a+r /sys/devices/platform/c240000.i2c/i2c-1/1-0041/hwmon/hwmon5/curr1_input
chmod a+r /sys/devices/platform/c240000.i2c/i2c-1/1-0041/hwmon/hwmon5/curr2_input
echo "done"
