"""
Demonstração das regras de substituição do HDB3 (000V e B00V) e da alternância das violações.

"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import codigos

SAIDA = 'resultados'


def ami(bits, pulso_anterior=-1):
    """AMI simples (só para comparação visual com o HDB3)."""
    s, ultimo = np.zeros(len(bits)), pulso_anterior
    for i, b in enumerate(bits):
        if b:
            ultimo = -ultimo
            s[i] = ultimo
    return s


def main():
    os.makedirs(SAIDA, exist_ok=True)
    bits = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    simb, info = codigos.hdb3(bits, retornar_info=True)

    # Resultado esperado : + - | +00+ | - + | -00- | +00+ | -
    esperado = np.array([1, -1, 1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, 0, 0, 1, -1])
    print('bits    :', ' '.join(str(b) for b in bits))
    print('HDB3    :', ' '.join({1: '+', -1: '-', 0: '0'}[int(v)] for v in simb))
    print('marcas  :', ' '.join(m if m else '.' for m in info['marca']))
    print('igual ao slide 25?', np.array_equal(simb, esperado))
    print('\nSubstituições (posição, padrão, sinal da violação):')
    for pos, padrao, v in info['substituicoes']:
        print(f'  posição {pos:2d}: {padrao}  V = {"+" if v > 0 else "-"}')
    print('Sinais das violações:', ['+' if v > 0 else '-' for v in info['sinais_violacoes']],
          '-> alternam' if np.all(np.diff(info['sinais_violacoes']) != 0) else '-> NÃO alternam')
    print('Decodifica de volta?', np.array_equal(codigos.decodificar_hdb3(simb), bits))

    n = len(bits)
    fig, axs = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True)
    for ax, s, nome in [(axs[0], ami(bits), 'AMI'), (axs[1], simb, 'HDB3')]:
        ax.step(np.r_[np.arange(n), n], np.r_[s, s[-1]], where='post', lw=2,
                color='black' if nome == 'AMI' else 'C3')
        ax.set_ylim(-1.7, 1.7)
        ax.set_yticks([-1, 0, 1], ['-A', '0', 'A'])
        ax.set_ylabel(nome)
        for k in range(n + 1):
            ax.axvline(k, color='C0', lw=0.4, ls='--')
    for k, b in enumerate(bits):
        axs[0].text(k + 0.5, 1.35, str(b), ha='center', color='C0')
        m = info['marca'][k]
        if m in ('B', 'V'):
            axs[1].text(k + 0.5, 0.35 if simb[k] < 0 else -0.6, m, ha='center',
                        color='C0', weight='bold')
    for pos, padrao, v in info['substituicoes']:
        axs[1].annotate(padrao, xy=(pos + 2, -1.55), ha='center', fontsize=9, color='gray')
    axs[1].set_xlabel('tempo (em bits)')
    axs[0].set_title('Exemplo do slide 25: o AMI fica parado nos zeros; o HDB3 troca cada 0000 por 000V ou B00V')
    fig.tight_layout()
    fig.savefig(os.path.join(SAIDA, 'hdb3_regras.png'), dpi=150)
    print(f'\nFigura salva em {SAIDA}/hdb3_regras.png')


if __name__ == '__main__':
    main()
