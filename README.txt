azTotMD 2.0 software (the Program)

The Program is developed with MS Visual Studio for Windows, but you can easily modify it for any platform.
The Program requires CUDA C++ compiler.

Use aztotmd.vcxproj project (the main unit is main.cu), not aztot_serial.vcxproj (the main unit is main.cpp).
The aztot_serial.vcxproj project is serial and is not supported now.

You could find executable file for Windows in /exec director

Directories "/case study" contain examples of input files for aztotmd.exe:
- case study 1 - 40'000 of Argon atoms controlled by radiative thermostat
- case study 2 -  4'000 abstract atoms with temperature-dependent force field and radiative thermostat