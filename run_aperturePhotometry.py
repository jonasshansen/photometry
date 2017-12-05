from aperturePhotometry import aperturePhotometry

run1 = aperturePhotometry('../run1/')
run2 = aperturePhotometry('../run2/')
run3 = aperturePhotometry('../run3/')

run4 = aperturePhotometry('../runRGB/', bias = True, datamethod = 'rgb',
                          radius = 25, bkginner = 30, bkgouter = 35,
                          stampsize = 36)
