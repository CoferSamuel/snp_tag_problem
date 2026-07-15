import matplotlib.pyplot as plt
import seaborn as sns

def configurar_tema(theme_name: str = 'default'):
    """
    Configura de manera global el estilo y paleta de seaborn/matplotlib 
    para las visualizaciones del proyecto.
    
    Parámetros:
    -----------
    theme_name : str
        Nombre del tema a cargar ('default' o 'epcc').
    """
    if theme_name == 'epcc':
        # Paleta corporativa EPCC / UEx:
        # Gris Oscuro EPCC, Amarillo EPCC, Verde Institucional UEx, Negro mate, Gris medio
        epcc_colors = [
            "#58585A",  # Gris EPCC (Principal)
            "#F1C40F",  # Amarillo EPCC (Secundario)
            "#008B3E",  # Verde UEx (Acento)
            "#222222",  # Negro mate (Acento oscuro)
            "#999999",  # Gris medio
            "#D4AC0D",  # Amarillo oscuro
            "#333333",  # Gris casi negro
        ]
        
        # Inyecta el tema base limpio y blanco (estilo whitegrid)
        sns.set_theme(
            style='whitegrid', 
            palette=epcc_colors,
            rc={
                'axes.edgecolor': '#cccccc',
                'grid.color': '#eeeeee',
                'font.family': 'sans-serif',
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'legend.fontsize': 10,
            }
        )
    else:
        # Tema por defecto (el que existía previamente en el repositorio de manera dispersa)
        sns.set_theme(style='whitegrid')
        sns.set_palette('muted')

