"""
Testes automáticos do simulador:
"""

import numpy as np
import codigos
import analise
import teoria


def testa_nrz():
    s = codigos.nrz_l([1, 0, 0, 1], A=2.0)
    assert np.array_equal(s, [2, -2, -2, 2])


def testa_manchester_slide():
    bits = [0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
    pulso = codigos.pulso_manchester(2)          # 2 amostras por bit = uma por metade
    y = codigos.formatar(codigos.nrz_l(bits), pulso)
    esperado = [-1, 1,  1, -1,  -1, 1,  -1, 1,  1, -1,  1, -1,  1, -1,  -1, 1,  -1, 1,  1, -1]
    assert np.array_equal(y, esperado)


def testa_manchester_teoria_area():
    """DEP teórica do Manchester integra para A^2 (= 1)."""
    f = np.linspace(-8000, 8000, 400001)
    S = teoria.dep_manchester(f, 1.0, 1e-3, sps=16)
    assert abs(np.trapezoid(S, f) - 1.0) < 0.01


def testa_rz_slide():
    bits = [0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
    y = codigos.formatar(codigos.rz_unipolar(bits), codigos.pulso_retangular(2, 0.5))
    esperado = [0, 0,  1, 0,  0, 0,  0, 0,  1, 0,  1, 0,  1, 0,  0, 0,  0, 0,  1, 0]
    assert np.array_equal(y, esperado)


def testa_rz_teoria_potencia():
    """Parte contínua + raias do RZ unipolar somam a potência média A^2 p1 / 2 (= 0,25)."""
    Tb, sps = 1e-3, 16
    f = np.linspace(-8000, 8000, 400001)
    cont, raias = teoria.dep_rz_unipolar(f, 1.0, Tb, 0.5, sps=sps)
    total = np.trapezoid(cont, f) + raias[0][1] + 2 * sum(p for fk, p in raias[1:] if fk < 8000)
    assert abs(total - 0.25) < 0.01


def testa_rz_polar_slide():
    bits = [0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
    y = codigos.formatar(codigos.nrz_l(bits), codigos.pulso_retangular(2, 0.5))
    esperado = [-1, 0,  1, 0,  -1, 0,  -1, 0,  1, 0,  1, 0,  1, 0,  -1, 0,  -1, 0,  1, 0]
    assert np.array_equal(y, esperado)


def testa_rz_polar_teoria():
    """RZ polar com p1 = 0,5: potência A^2/2 e nenhuma raia."""
    f = np.linspace(-8000, 8000, 400001)
    cont, raias = teoria.dep_rz_polar(f, 1.0, 1e-3, 0.5, sps=16)
    assert abs(np.trapezoid(cont, f) - 0.5) < 0.01
    assert max(p for _, p in raias) < 1e-12


def testa_ami_slide():
    bits = [0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
    assert np.array_equal(codigos.ami(bits), [0, 1, 0, 0, -1, 1, -1, 0, 0, 1])


def testa_ami_sem_dc():
    """A soma acumulada dos símbolos AMI fica sempre entre 0 e 1 ."""
    bits = np.random.default_rng(2).integers(0, 2, 20000)
    soma = np.cumsum(codigos.ami(bits))
    assert soma.min() >= 0 and soma.max() <= 1


def testa_hdb3_slide():
    bits = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    esperado = [1, -1, 1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, 0, 0, 1, -1]
    assert np.array_equal(codigos.hdb3(bits), esperado)


def testa_hdb3_aleatorio():
    rng = np.random.default_rng(0)
    for p1 in (0.1, 0.3, 0.5, 0.9):       
        bits = (rng.random(20000) < p1).astype(np.int8)
        s, info = codigos.hdb3(bits, retornar_info=True)
        # 1) decodifica de volta sem erro
        assert np.array_equal(codigos.decodificar_hdb3(s), bits)
        # 2) nunca mais de 3 zeros seguidos
        zeros = 0
        for v in s:
            zeros = zeros + 1 if v == 0 else 0
            assert zeros <= 3
        # 3) violações alternam de sinal
        sv = info['sinais_violacoes']
        assert np.all(np.diff(sv) != 0)
        # 4) soma acumulada limitada (sem deriva de DC)
        assert np.abs(np.cumsum(s)).max() <= 3


def testa_potencia_welch():
    """A área da DEP de Welch tem de bater com a potência média do sinal."""
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, 50000)
    y = codigos.formatar(codigos.nrz_l(bits), codigos.pulso_retangular(16))
    f, P = analise.dep_welch(y, 16000, 16)
    area = np.trapezoid(P, f)
    assert abs(area - analise.potencia_media(y)) < 0.02


def testa_banda90_rz():
    """Banda de 90% medida do RZ unipolar bate com a teórica (DC medido à parte)."""
    from simular import banda90_de_dep
    Rb, sps = 1000.0, 16
    Tb, fs = 1 / Rb, sps * Rb
    bits = np.random.default_rng(3).integers(0, 2, 131072)
    y = codigos.formatar(codigos.rz_unipolar(bits), codigos.pulso_retangular(sps, 0.5))
    dc = analise.nivel_dc(y)
    f, Pac = analise.dep_welch(y - dc, fs, sps)
    b90_med = analise.banda_fracao(f, Pac, 0.9, potencia_dc=dc ** 2)
    ff = np.linspace(0, fs / 2, 20001)
    cont, raias = teoria.dep_rz_unipolar(ff, 1.0, Tb, 0.5, sps=sps)
    assert abs(b90_med - banda90_de_dep(cont, raias, ff)) < 0.05 * Rb


def testa_teoria_area():
    """DEP teórica do NRZ polar integra para A^2 (= 1)."""
    f = np.linspace(-8000, 8000, 400001)
    cont, raia = teoria.dep_nrz_polar(f, 1.0, 1e-3, 0.5, sps=16)
    assert abs(np.trapezoid(cont, f) - 1.0) < 0.01 and raia == 0


if __name__ == '__main__':
    for nome, fn in list(globals().items()):
        if nome.startswith('testa_'):
            fn()
            print(f'ok  {nome}')
    print('Todos os testes passaram.')