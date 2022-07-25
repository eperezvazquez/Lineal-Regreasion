import pandas as pd


class Helpers():
    @classmethod
    # función para convertir región a numérico
    def conv_region(self,region_name):
        if region_name == 'southwest':
            return 1
        elif region_name == 'southeast':
            return 2
        elif region_name == 'northwest':
            return 3
        elif region_name == 'northeast':
            return 4
        else:
            return 'región sin determinar'