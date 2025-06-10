import os
from cairosvg import svg2png

logo = "53b1fbc5-2ca8-4386-b5d2-4daec09fd335.svg"

def export_svg(SVG,output,size:tuple):
    """
    Export SVG to PNG
    :param SVG: SVG file path
    :param output: Output file path
    :param size: Size of the output image
    """
    svg2png(url=SVG, write_to=output, output_width=size[0], output_height=size[1])


for i in [16,48,128]:
    export_svg(logo, os.path.join( f"logo-{i}.png"), (i,i))