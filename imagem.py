"""
Geração de bits a partir de uma imagem.

Modo 'cinza': cada pixel (0 a 255) vira 8 bits (mais significativo primeiro).
Modo 'pb'   : cada pixel vira 1 bit (1 se claro, 0 se escuro).

Imagens têm regiões uniformes (fundo branco, áreas pretas), que geram sequencias de bits iguais. Isso é justamente o que testa a 
capacidade de cada código de manter transições para o relógio.
"""

import numpy as np
from PIL import Image, ImageDraw


def imagem_sintetica(tamanho=(128, 128)):
    """Imagem de teste reprodutível: fundo branco, áreas pretas, degradê e círculo."""
    w, h = tamanho
    img = Image.new('L', (w, h), 255)                 # fundo branco
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w // 3, h // 3], fill=0)       # quadrado preto
    for x in range(w // 2, w):                        # faixa em degradê
        tom = int(255 * (x - w // 2) / (w // 2))
        d.line([(x, h // 2), (x, h - 1)], fill=tom)
    d.ellipse([w // 6, h // 2, w // 2, h - h // 8], fill=60)   # círculo cinza escuro
    return img


def bits_da_imagem(caminho=None, tamanho=(128, 128), modo='cinza', limiar=128):
    """
    Devolve (bits, pixels).

    caminho : arquivo de imagem (png, jpg...). Se None, usa a imagem sintética.
    tamanho : a imagem é redimensionada para (largura, altura), para controlar o número de bits e o tempo de simulação.
    """
    if caminho is None:
        img = imagem_sintetica(tamanho)
    else:
        img = Image.open(caminho).convert('L').resize(tamanho)
    px = np.asarray(img, dtype=np.uint8)
    if modo == 'pb':
        bits = (px.flatten() >= limiar).astype(np.int8)
    else:
        bits = np.unpackbits(px.flatten()).astype(np.int8)
    return bits, px
