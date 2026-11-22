"""
Medidas feitas sobre a forma de onda simulada.

- DEP estimada por Welch
- nível DC
- conteúdo espectral na frequência de símbolo (f = Rb)
- largura de banda que contém 90% da potência
- maior sequência sem transições (run length) e densidade de transições
"""

import numpy as np
from scipy.signal import welch
from scipy.integrate import cumulative_trapezoid


def dep_welch(x, fs, sps, bits_por_segmento=64):
    """
    DEP unilateral (W/Hz) pelo método de Welch.

    Cada segmento tem `bits_por_segmento` bits, então a resolução em frequência é Rb / bits_por_segmento.

    detrend=False é essencial: o padrão do scipy ('constant') subtrai a média de cada segmento e apagaria a componente DC que queremos medir.
    """
    nperseg = bits_por_segmento * sps
    f, P = welch(x, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg // 2, detrend=False, return_onesided=True, scaling='density')
    return f, P


def nivel_dc(x):
    return float(np.mean(x))


def potencia_media(x):
    return float(np.mean(x ** 2))


def banda_fracao(f, P, fracao=0.90, potencia_dc=0.0):
    """
    Menor frequência B tal que a potência entre 0 e B é fracao do total.

    potencia_dc : potência da raia em DC, somada à parte em f = 0.
        Para sinais com DC (NRZ unipolar, RZ, NRZ-L com a imagem), passe aqui média² e use em P a DEP do sinal sem a média (x - média). 
        O motivo: no Welch a raia em DC fica na borda f = 0, e a integral por trapézio conta só metade dela, o que subestima o DC e desloca a banda.
    """
    # Soma por RETÂNGULOS: cada ponto do Welch representa a faixa [f - df/2, f + df/2].
    # (O trapézio conta só metade do 1º ponto, em f = 0, e com a imagem esse ponto
    #  concentra muita potência de baixa frequência -> a banda saía deslocada.)
    df = f[1] - f[0]
    acumulada = potencia_dc + np.cumsum(P) * df
    acumulada /= acumulada[-1]
    return float(np.interp(fracao, acumulada, f + df / 2))


def conteudo_em_rb(f, P, Rb):
    """
    Conteúdo espectral em f = Rb.

    Devolve:
      rel_pico_db : S(Rb) em dB em relação ao máximo da DEP
                    (valores muito negativos indicam um NULO em Rb)
      raia_db     : S(Rb) em relação à mediana da DEP vizinha (0,8Rb a 1,2Rb,
                    sem o próprio ponto). Acima de ~10 dB indica uma RAIA
                    (componente discreta útil para extrair o relógio).
    """
    s_rb = np.interp(Rb, f, P)
    rel_pico_db = 10 * np.log10(s_rb / P.max() + 1e-300)
    viz = (np.abs(f - Rb) > 2 * (f[1] - f[0])) & (np.abs(f - Rb) < 0.2 * Rb)
    raia_db = 10 * np.log10(s_rb / (np.median(P[viz]) + 1e-300) + 1e-300)
    return float(rel_pico_db), float(raia_db)


def run_length(x, sps):
    """
    Maior trecho SEM TRANSIÇÃO da forma de onda, em períodos de bit.
    Funciona para qualquer formato de pulso (NRZ, RZ, Manchester...),
    porque olha a forma de onda amostrada e não os bits.

    Também devolve a densidade de transições (transições por bit).
    """
    mudancas = np.flatnonzero(np.diff(x) != 0)
    if len(mudancas) == 0:
        return len(x) / sps, 0.0
    limites = np.r_[-1, mudancas, len(x) - 1]
    trechos = np.diff(limites)
    return float(trechos.max() / sps), float(len(mudancas) / (len(x) / sps))


def erro_relogio_toleravel(run_max_bits):
    """
    Relação entre run length e recuperação do relógio.

    O receptor corrige a fase do relógio a cada transição. Se o relógio local
    tem erro relativo de frequência eps, depois de L bits sem transição o erro
    de tempo acumulado é eps*L*Tb. Para continuar amostrando dentro do bit,
    é preciso eps*L*Tb < Tb/2, ou seja, eps < 1/(2L).
    """
    return 1.0 / (2.0 * run_max_bits)