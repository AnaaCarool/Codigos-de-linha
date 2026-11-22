"""
Codificadores de linha.

Cada codificador recebe uma sequência de bits e devolve a sequência de amplitudes dos símbolos a_k (um valor por bit). A forma de onda é montada
depois, em `formatar`, multiplicando cada a_k por um pulso p(t):

    y(t) = soma_k  a_k * p(t - k*Tb)        (modelo PAM)

Assim, o codificador decide a altura e o pulso decide o formato.
"""

import numpy as np


# ---------------------------------------------------------------------------
# NRZ-L (polar): 1 -> +A, 0 -> -A, pulso ocupando o bit inteiro
# ---------------------------------------------------------------------------
def nrz_l(bits, A=1.0):
    bits = np.asarray(bits, dtype=np.int8)
    return A * (2.0 * bits - 1.0)


# ---------------------------------------------------------------------------
# RZ unipolar: 1 -> +A na 1ª metade do bit e 0 na 2ª; 0 -> sem pulso
# ---------------------------------------------------------------------------
def rz_unipolar(bits, A=1.0):
    """
    O "retorno a zero" no meio do bit vem do pulso de meia largura (pulso_retangular(sps, 0.5))
     """
    bits = np.asarray(bits, dtype=np.int8)
    return A * bits.astype(float)


# ---------------------------------------------------------------------------
# AMI: bit 0 -> sem pulso (0); bit 1 -> pulso que alterna de sinal (+A, -A).
# ---------------------------------------------------------------------------
def ami(bits, A=1.0, pulso_anterior=-1):
    bits = np.asarray(bits, dtype=np.int8)
    simb = np.zeros(len(bits))
    ultimo = pulso_anterior
    for i, b in enumerate(bits):
        if b == 1:
            ultimo = -ultimo      
            simb[i] = ultimo
    return A * simb


# ---------------------------------------------------------------------------
# HDB3 (High Density Bipolar 3-zeros)
# ---------------------------------------------------------------------------
def hdb3(bits, A=1.0, pulso_anterior=-1, retornar_info=False):
    """
    Codificador HDB3.

    Base AMI: bit 0 -> sem pulso; bit 1 -> pulso que alterna de sinal.
    Cada bloco de 4 zeros seguidos é trocado por:
        000V  se o número de uns desde a última substituição é ÍMPAR
        B00V  se esse número é PAR (zero conta como par)
    onde
        V (violação) = pulso com o mesmo sinal do último pulso enviado
        B            = pulso normal, com sinal OPOSTO ao último pulso
    Essa escolha faz as violações alternarem de sinal (sem componente DC).

    Parâmetros
    ----------
    bits : sequência de 0/1
    A : amplitude dos pulsos
    pulso_anterior : sinal (+1/-1) do "último pulso" antes do início.
        Com -1, o primeiro bit 1 sai positivo (como no slide 25).
    retornar_info : se True, devolve também um dicionário com as
        substituições feitas (útil para mostrar as regras e conferir a
        alternância das violações).
    """
    bits = np.asarray(bits, dtype=np.int8)
    n = len(bits)
    simb = np.zeros(n)            # sinais (-1, 0, +1) de cada bit
    marca = np.full(n, '', dtype='<U1')   # '1' (pulso normal), 'B' ou 'V'

    ultimo = pulso_anterior       # sinal do último pulso enviado
    uns_desde_subst = 0           # contagem de uns desde a última substituição
    substituicoes = []            # (posição, padrão, sinal da violação)

    i = 0
    while i < n:
        if bits[i] == 1:
            # Regra AMI: cada 1 tem o sinal oposto ao do último pulso
            ultimo = -ultimo
            simb[i] = ultimo
            marca[i] = '1'
            uns_desde_subst += 1
            i += 1
        elif i + 3 < n and not bits[i:i + 4].any():
            # Bloco de 4 zeros: substituir
            if uns_desde_subst % 2 == 1:
                # ÍMPAR -> 000V : só a violação, igual ao último pulso
                v = ultimo
                padrao = '000V'
            else:
                # PAR -> B00V : B inverte, e o V copia o sinal do B
                b = -ultimo
                simb[i] = b
                marca[i] = 'B'
                v = b
                padrao = 'B00V'
            simb[i + 3] = v
            marca[i + 3] = 'V'
            ultimo = v                # o próximo 1 alterna em relação ao V
            uns_desde_subst = 0       # a contagem recomeça
            substituicoes.append((i, padrao, int(v)))
            i += 4
        else:
            # Zero "comum" (menos de 4 seguidos): sem pulso
            i += 1

    simb = A * simb
    if retornar_info:
        info = {
            'substituicoes': substituicoes,
            'marca': marca,
            'sinais_violacoes': np.array([s[2] for s in substituicoes]),
        }
        return simb, info
    return simb


def decodificar_hdb3(simb, pulso_anterior=-1):
    """
    Decodificador HDB3 (usado para validar o codificador).

    Qualquer pulso vale 1, exceto quando é uma VIOLAÇÃO (mesmo sinal do pulso
    anterior): aí ele e as 3 posições anteriores eram zeros (000V ou B00V).
    """
    s = np.sign(np.asarray(simb)).astype(int)
    bits = (s != 0).astype(np.int8)
    ultimo = pulso_anterior
    for i, v in enumerate(s):
        if v == 0:
            continue
        if v == ultimo:                 # violação detectada
            bits[max(0, i - 3):i + 1] = 0
        ultimo = v
    return bits


# ---------------------------------------------------------------------------
# Montagem da forma de onda (modelo PAM)
# ---------------------------------------------------------------------------
def pulso_retangular(sps, largura=1.0):
    """
    Pulso retangular de altura 1 amostrado com `sps` amostras por bit. largura = 1.0 -> NRZ (bit inteiro); largura = 0.5 -> RZ (meio bit).
    """
    m = int(round(largura * sps))
    return np.r_[np.ones(m), np.zeros(sps - m)]


def pulso_manchester(sps):
    """
    Pulso do Manchester: +1 na 1ª metade do bit, -1 na 2ª.
    Com a_k = +A (bit 1) fica +A -> -A; com a_k = -A (bit 0) fica -A -> +A.
    """
    meio = sps // 2
    return np.r_[np.ones(meio), -np.ones(sps - meio)]


def formatar(simbolos, pulso):
    """Forma de onda y = soma a_k p(t - kTb): repete o pulso escalado por a_k."""
    return np.kron(np.asarray(simbolos, dtype=float), pulso)


#   'codificar': função bits -> amplitudes a_k
#   'pulso'    : função sps -> vetor com o formato do pulso p(t) amostrado
CODIGOS = {
    'NRZ-L': {'codificar': nrz_l, 'pulso': lambda sps: pulso_retangular(sps, 1.0)},

    'RZ':    {'codificar': rz_unipolar, 'pulso': lambda sps: pulso_retangular(sps, 0.5)},

    'RZ-polar': {'codificar': nrz_l, 'pulso': lambda sps: pulso_retangular(sps, 0.5)},
    
    'Manchester': {'codificar': nrz_l, 'pulso': pulso_manchester},
   
    'AMI':   {'codificar': ami,   'pulso': lambda sps: pulso_retangular(sps, 1.0)},

    'HDB3':  {'codificar': hdb3,  'pulso': lambda sps: pulso_retangular(sps, 1.0)},

}