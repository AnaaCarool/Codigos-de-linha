"""
DEPs teóricas (bilaterais, em W/Hz).

    S_y(f) = |P(f)|^2 * S_x(f)
    S_x(f) = (1/Tb) * [ R0 + 2 * soma_{n>=1} R_n cos(2 pi n f Tb) ]

"""

import numpy as np


def P2_retangular(f, Tb, largura=1.0, sps=None):
    """
    |P(f)|^2 de um pulso retangular de altura 1 e duração largura*Tb.

    sps=None  -> pulso contínuo: (tau * sinc(f tau))^2
    sps=int   -> pulso amostrado com sps amostras por bit .
    """
    f = np.asarray(f, dtype=float)
    if sps is None:
        tau = largura * Tb
        return (tau * np.sinc(f * tau)) ** 2
    Ts = Tb / sps
    m = int(round(largura * sps))
    num = np.sin(np.pi * f * m * Ts)
    den = np.sin(np.pi * f * Ts)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where(np.abs(den) < 1e-12, m * Ts, Ts * num / den)
    return P ** 2


def dep_nrz_polar(f, A, Tb, p1=0.5, sps=None):
    """
    NRZ polar com bits independentes e P(bit=1) = p1.

    Média dos símbolos  m = A(2 p1 - 1)
    Variância           s2 = 4 p1 (1 - p1) A^2
    Parte contínua:     S(f) = s2 / Tb * |P(f)|^2      (= A^2 Tb sinc^2(f Tb) se p1 = 0,5)
    Raia em DC:         potência m^2 (devolvida à parte, pois é um impulso)
    """
    s2 = 4 * p1 * (1 - p1) * A ** 2
    cont = s2 / Tb * P2_retangular(f, Tb, 1.0, sps)
    potencia_raia_dc = (A * (2 * p1 - 1)) ** 2
    return cont, potencia_raia_dc


def dep_ami(f, A, Tb, largura=1.0, sps=None):
    """
    AMI com bits equiprováveis e independentes (aproximação teórica do HDB3).
    R0 = A^2/2, R1 = -A^2/4, Rn = 0 (n >= 2)
    S(f) = (A^2/Tb) |P(f)|^2 sen^2(pi f Tb)
    """
    return (A ** 2 / Tb) * P2_retangular(f, Tb, largura, sps) * np.sin(np.pi * f * Tb) ** 2


def dep_rz_unipolar(f, A, Tb, p1=0.5, sps=None):
    """
    RZ unipolar (pulso de meio bit), bits independentes com P(bit=1) = p1.

    Símbolos a_k = 0 ou A:
        média      m  = A p1          (diferente de zero -> RAIAS)
        variância  s2 = A^2 p1 (1 - p1)
    Parte contínua:  S(f) = s2 / Tb * |P(f)|^2
                     (= A^2 Tb/16 * sinc^2(f Tb/2) se p1 = 0,5)
    Raias em f = k/Tb (k = 0, 1, 2...), cada uma com potência
                     m^2 |P(k/Tb)|^2 / Tb^2
    As raias pares (2Rb, 4Rb...) caem nos zeros do sinc e somem;
    a raia em Rb é a que serve de RELÓGIO.

    Devolve (parte contínua bilateral, lista de raias [(freq, potência)] com k >= 0),
    no mesmo formato de dep_pelos_Rn.
    """
    f = np.asarray(f, dtype=float)
    s2 = A ** 2 * p1 * (1 - p1)
    m = A * p1
    cont = s2 / Tb * P2_retangular(f, Tb, 0.5, sps)
    fk = np.arange(0, int(np.max(f) * Tb) + 1) / Tb
    pk = m ** 2 * P2_retangular(fk, Tb, 0.5, sps) / Tb ** 2
    return cont, [(float(a), float(b)) for a, b in zip(fk, pk)]


def dep_rz_polar(f, A, Tb, p1=0.5, sps=None):
    """
    RZ polar (pulso de meio bit), bits independentes com P(bit=1) = p1.

    Símbolos a_k = +A ou -A (os mesmos do NRZ-L):
        média      m  = A (2 p1 - 1)     (zero se p1 = 0,5 -> sem raias)
        variância  s2 = 4 p1 (1 - p1) A^2
    Parte contínua:  S(f) = s2 / Tb * |P(f)|^2   (= A^2 Tb/4 * sinc^2(f Tb/2) se p1 = 0,5)
    Raias em f = k/Tb com potência m^2 |P(k/Tb)|^2 / Tb^2 (só aparecem se p1 != 0,5).

    Comparado ao RZ unipolar: MESMO pulso, outra estatística. Com p1 = 0,5
    a média é zero e a raia em Rb desaparece, apesar de o sinal ter
    transição em todo bit.
    """
    f = np.asarray(f, dtype=float)
    s2 = 4 * p1 * (1 - p1) * A ** 2
    m = A * (2 * p1 - 1)
    cont = s2 / Tb * P2_retangular(f, Tb, 0.5, sps)
    fk = np.arange(0, int(np.max(f) * Tb) + 1) / Tb
    pk = m ** 2 * P2_retangular(fk, Tb, 0.5, sps) / Tb ** 2
    return cont, [(float(a), float(b)) for a, b in zip(fk, pk)]


def dep_manchester(f, A, Tb, sps=None):
    """
    Manchester com bits equiprováveis e independentes.
    Estatística igual à do NRZ polar: R0 = A^2, Rn = 0 (n >= 1) -> S_x(f) = A^2 / Tb.
    Muda só o pulso (+1 na 1ª metade, -1 na 2ª):
        |P(f)|^2 = Tb^2 sinc^2(f Tb/2) sen^2(pi f Tb/2)
        S(f)     = A^2 Tb sinc^2(f Tb/2) sen^2(pi f Tb/2)
    sps=int -> usa o pulso AMOSTRADO (o que o simulador gera de fato).
    """
    f = np.asarray(f, dtype=float)
    if sps is None:
        P2 = Tb ** 2 * np.sinc(f * Tb / 2) ** 2 * np.sin(np.pi * f * Tb / 2) ** 2
    else:
        meio = sps // 2
        pulso = np.r_[np.ones(meio), -np.ones(sps - meio)]
        P2 = P2_pulso_amostrado(f, pulso, Tb / sps)
    return (A ** 2 / Tb) * P2


def autocovariancia(simb, max_lag):
    """C_n = E[(a_k - m)(a_{k+n} - m)] estimada da sequência, n = 0..max_lag."""
    c = np.asarray(simb, dtype=float) - np.mean(simb)
    n = len(c)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(c, nfft)
    acf = np.fft.irfft(np.abs(X) ** 2, nfft)[:max_lag + 1] / n
    return acf


def P2_pulso_amostrado(f, pulso, Ts):
    """
    |P(f)|^2 de QUALQUER pulso amostrado (vetor `pulso`, período de amostragem Ts):
        P(f) = Ts * soma_n pulso[n] * exp(-j 2 pi f n Ts)
    Serve para NRZ, RZ, Manchester... (para o retangular coincide com P2_retangular).
    """
    f = np.asarray(f, dtype=float)
    n = np.arange(len(pulso))
    P = Ts * np.exp(-2j * np.pi * np.outer(f, n) * Ts) @ np.asarray(pulso, dtype=float)
    return np.abs(P) ** 2


def dep_pelos_Rn(f, simb, Tb, pulso, max_lag=64):
    """
    DEP 'semi-analítica': a fórmula da aula, mas com os R_n MEDIDOS na
    sequência de símbolos real (em vez de supor bits independentes).

    Separa a média m dos símbolos (que gera raias) da parte aleatória
    (autocovariância C_n):
        contínua: |P(f)|^2 / Tb * [C0 + 2 soma w_n C_n cos(2 pi n f Tb)]
        raias em f = k/Tb com potência m^2 |P(k/Tb)|^2 / Tb^2
    w_n é uma janela triangular (Bartlett), que mantém a estimativa >= 0.
    Use max_lag igual ao nº de bits por segmento do Welch, para as duas
    estimativas terem resolução parecida.

    Se esta curva coincide com o Welch e a teórica 'ideal' não, a diferença
    vem da ESTATÍSTICA DOS BITS (bits da imagem não são independentes nem
    equiprováveis), e não de erro na simulação.

    Devolve (parte contínua bilateral, lista de raias [(freq, potência)] com k >= 0).
    """
    Ts = Tb / len(pulso)
    C = autocovariancia(simb, max_lag)
    n = np.arange(1, max_lag + 1)
    w = 1 - n / (max_lag + 1)
    f = np.asarray(f, dtype=float)
    Sx = C[0] + 2 * np.cos(2 * np.pi * np.outer(f, n) * Tb) @ (w * C[1:])
    cont = P2_pulso_amostrado(f, pulso, Ts) * Sx / Tb
    m = float(np.mean(simb))
    fk = np.arange(0, int(f.max() * Tb) + 1) / Tb
    pk = m ** 2 * P2_pulso_amostrado(fk, pulso, Ts) / Tb ** 2
    raias = [(float(a), float(b)) for a, b in zip(fk, pk)]
    return cont, raias