"""
Uso básico:
    python simular.py                       # imagem sintética + bits aleatórios
    python simular.py --imagem foto.png     # sua imagem
    python simular.py --modo pb             # 1 bit por pixel (sequências bem longas)
    python simular.py --sps 32 --segmento 128 --formato-hdb3 rz

Para cada FONTE de bits (aleatória = controle, e imagem) e cada CÓDIGO:
  1. codifica e monta a forma de onda;
  2. estima a DEP por Welch e compara com a teórica;
  3. mede DC, conteúdo em Rb, banda de 90% e run length;
  4. salva figuras e a tabela em ./resultados.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

import codigos
import analise
import teoria
from imagem import bits_da_imagem

ROTULO_FONTE = {'aleatoria': 'bits aleatórios (controle)', 'imagem': 'bits da imagem'}


def parse_args():
    p = argparse.ArgumentParser(description='Simulação de códigos de linha')
    p.add_argument('--imagem', default=None, help='arquivo de imagem (padrão: imagem sintética)')
    p.add_argument('--modo', choices=['cinza', 'pb'], default='cinza', help='8 bits/pixel ou 1 bit/pixel')
    p.add_argument('--tamanho', type=int, default=128, help='lado da imagem após redimensionar (pixels)')
    p.add_argument('--rb', type=float, default=1000.0, help='taxa de bits Rb (bit/s)')
    p.add_argument('--sps', type=int, default=16, help='amostras por bit')
    p.add_argument('--A', type=float, default=1.0, help='amplitude dos pulsos (V)')
    p.add_argument('--segmento', type=int, default=64, help='bits por segmento do Welch')
    p.add_argument('--formato-hdb3', choices=['nrz', 'rz'], default='nrz', help='largura do pulso do HDB3 e do AMI')
    p.add_argument('--codigos', nargs='+', default=list(codigos.CODIGOS),
                   help='códigos a simular (padrão: todos)')
    p.add_argument('--semente', type=int, default=1, help='semente dos bits aleatórios')
    p.add_argument('--saida', default='resultados')
    return p.parse_args()


def dep_teorica_ideal(nome, f, A, Tb, largura, sps, p1):
    if nome == 'NRZ-L':
        cont, raia = teoria.dep_nrz_polar(f, A, Tb, p1=p1, sps=sps)
        return cont, raia, f'teórica NRZ polar (bits indep., p1 = {p1:.2f})'
    if nome == 'RZ':
        # aqui a 'raia' é uma LISTA [(freq, potência)]: DC, Rb, 3Rb...
        cont, raias = teoria.dep_rz_unipolar(f, A, Tb, p1=p1, sps=sps)
        return cont, raias, f'teórica RZ unipolar (bits indep., p1 = {p1:.2f})'
    if nome == 'RZ-polar':
        cont, raias = teoria.dep_rz_polar(f, A, Tb, p1=p1, sps=sps)
        return cont, raias, f'teórica RZ polar (bits indep., p1 = {p1:.2f})'
    if nome == 'Manchester':
        return teoria.dep_manchester(f, A, Tb, sps), 0.0, 'teórica Manchester (bits indep.)'
    if nome == 'AMI':
        return teoria.dep_ami(f, A, Tb, largura, sps), 0.0, 'teórica AMI (bits indep.)'
    if nome == 'HDB3':
        return teoria.dep_ami(f, A, Tb, largura, sps), 0.0, 'teórica AMI (aproximação do HDB3)'
    raise ValueError(f'sem DEP teórica cadastrada para {nome}')


def lista_de_raias(raias):
    if np.isscalar(raias):
        return [(0.0, float(raias))]
    return list(raias)


def banda90_de_dep(cont, raias, f):
    acum = cumulative_trapezoid(2 * cont, f, initial=0.0)
    for fk, pk in lista_de_raias(raias):
        acum = acum + np.where(f >= fk, pk if fk == 0 else 2 * pk, 0.0)
    acum /= acum[-1]
    return float(np.interp(0.9, acum, f))


def figura_dep(f, Pw, cont_ideal, raia_ideal, rot_ideal, cont_rn, raias_rn, dc_med,
               Rb, A, Tb, titulo, caminho):
    """Welch x teórica ideal x teórica com R_n medidos (escala em dB)."""
    norm = A ** 2 * Tb
    m = f > 0
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(f[m] / Rb, 10 * np.log10(Pw[m] / norm), label='simulada (Welch)',
            color='tab:green', lw=1.6)                
    ax.plot(f[m] / Rb, 10 * np.log10(2 * cont_ideal[m] / norm + 1e-12), '--', label=rot_ideal,
            color='tab:orange', lw=1.6)
    ax.plot(f[m] / Rb, 10 * np.log10(2 * cont_rn[m] / norm + 1e-12), ':',
            label='teórica com $R_n$ medidos', color='black', lw=1.2)   
    ax.set_xlim(0, 4)
    ax.set_ylim(-50, 8)
    ax.set_xlabel('f / Rb')
    ax.set_ylabel('DEP unilateral / (A² Tb)  [dB]')
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    for fk, pk in raias_rn[1:]:                       # raias fora do DC 
        if pk > 1e-3 * A ** 2 and fk / Rb <= 4:
            ax.axvline(fk / Rb, color='tab:purple', lw=1, ls='-.')
            ax.text(fk / Rb, 5, f' raia {pk:.3f} W', color='tab:purple', fontsize=7)
    raia_dc = raias_rn[0][1]
    raia_ideal = lista_de_raias(raia_ideal)[0][1]     # só a raia em DC vai para o texto
    if raia_dc > 1e-3 * A ** 2:
        ax.text(0.02, 0.04,
                f'Raia em DC (impulso, fora da escala):\n'
                f'  teórica ideal: {raia_ideal:.3f} W   com R_n: {raia_dc:.3f} W\n'
                f'  medida (média²): {dc_med ** 2:.3f} W',
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round', fc='white', alpha=0.85))
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def figura_formas_de_onda(sps, A, pulsos, caminho):
    """
    No HDB3, as letras B e V marcam as substituições.
    """
    bits = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1])
    n = len(bits)
    t = np.arange(n * sps) / sps
    nomes = list(pulsos)

    fig, axs = plt.subplots(len(nomes), 1, figsize=(10, 1.25 * len(nomes) + 0.8), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, nome in zip(axs, nomes):
        if nome == 'HDB3':
            simb, info = codigos.hdb3(bits, A, retornar_info=True)
        else:
            simb, info = codigos.CODIGOS[nome]['codificar'](bits, A), None
        y = codigos.formatar(simb, pulsos[nome])
        ax.step(np.append(t, n), np.append(y, y[-1]), where='post', lw=1.6)
        ax.set_ylim(-1.7 * A, 1.7 * A)
        ax.set_yticks([-A, 0, A])
        ax.set_ylabel(nome, rotation=0, ha='right', va='center', fontsize=9)
        ax.grid(alpha=0.3, axis='y')
        for k in range(n + 1):
            ax.axvline(k, color='gray', lw=0.3)
        if info is not None:
            for k in range(n):
                if info['marca'][k] in ('B', 'V'):
                    ax.text(k + 0.5, 1.25 * A if simb[k] > 0 else -1.6 * A, info['marca'][k],
                            ha='center', color='tab:red', fontsize=9, weight='bold')
    for k, b in enumerate(bits):
        axs[0].text(k + 0.5, 1.95 * A, str(b), ha='center', fontsize=9)
    axs[-1].set_xlabel('tempo (em bits)')
    fig.suptitle('Mesmos bits em todos os códigos (bits no topo; B e V = substituições do HDB3)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def figura_imagem(px, n_bits, modo, caminho):
    fig, ax = plt.subplots(figsize=(3.2, 3.4))
    ax.imshow(px, cmap='gray', vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Imagem ({modo}, {n_bits} bits)', fontsize=9)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def main():
    a = parse_args()
    os.makedirs(a.saida, exist_ok=True)
    rng = np.random.default_rng(a.semente)
    Rb, sps, A = a.rb, a.sps, a.A
    Tb, fs = 1.0 / Rb, sps * Rb

    # Formato do pulso de cada código (o HDB3 pode ser trocado para RZ pela linha de comando)
    pulsos = {nome: codigos.CODIGOS[nome]['pulso'](sps) for nome in a.codigos}
    if a.formato_hdb3 == 'rz':                  # vale para HDB3 e AMI
        for nome in ('HDB3', 'AMI'):
            if nome in pulsos:
                pulsos[nome] = codigos.pulso_retangular(sps, 0.5)
    largura = {nome: float(np.count_nonzero(p)) / sps for nome, p in pulsos.items()}

    bits_img, px = bits_da_imagem(a.imagem, (a.tamanho, a.tamanho), a.modo)
    fontes = {
        'aleatoria': rng.integers(0, 2, len(bits_img)).astype(np.int8),   
        'imagem': bits_img,
    }
    figura_imagem(px, len(bits_img), a.modo, os.path.join(a.saida, 'imagem_usada.png'))

    linhas = []
    ff = np.linspace(0, fs / 2, 20001)          
    for nome_fonte, bits in fontes.items():
        p1 = float(bits.mean())
        for nome in a.codigos:
            cod = codigos.CODIGOS[nome]['codificar']
            extra = ''
            if nome == 'HDB3':
                simb, info = cod(bits, A, retornar_info=True)
                # decodifica de volta e confere a alternância das violações
                assert np.array_equal(codigos.decodificar_hdb3(simb), bits), 'HDB3 não decodificou certo!'
                sv = info['sinais_violacoes']
                alterna = bool(np.all(np.diff(sv) != 0)) if len(sv) > 1 else True
                n000v = sum(1 for s in info['substituicoes'] if s[1] == '000V')
                nb00v = len(info['substituicoes']) - n000v
                extra = (f'substituições: {n000v}x 000V e {nb00v}x B00V; '
                         f'violações alternam de sinal: {"sim" if alterna else "NÃO"}')
            else:
                simb = cod(bits, A)
            y = codigos.formatar(simb, pulsos[nome])

            # Medidas na forma de onda simulada
            f, Pw = analise.dep_welch(y, fs, sps, a.segmento)
            dc = analise.nivel_dc(y)
            pot = analise.potencia_media(y)
            rel_db, raia_db = analise.conteudo_em_rb(f, Pw, Rb)
            # Banda de 90%: DC medido pela média + Welch do sinal sem a média
            _, Pac = analise.dep_welch(y - dc, fs, sps, a.segmento)
            b90 = analise.banda_fracao(f, Pac, 0.90, potencia_dc=dc ** 2)
            run_max, dens = analise.run_length(y, sps)

            # Previsões teóricas
            p1_teo = p1 if nome in ('NRZ-L', 'RZ', 'RZ-polar') else 0.5
            cont_i, raia_i, rot_i = dep_teorica_ideal(nome, f, A, Tb, largura[nome], sps, p1_teo)
            cont_r, raias_r = teoria.dep_pelos_Rn(f, simb, Tb, pulsos[nome], max_lag=a.segmento)
            ci, ri, _ = dep_teorica_ideal(nome, ff, A, Tb, largura[nome], sps, p1_teo)
            cr, rr = teoria.dep_pelos_Rn(ff, simb, Tb, pulsos[nome], max_lag=a.segmento)
            b90_teo = banda90_de_dep(ci, ri, ff)
            b90_rn = banda90_de_dep(cr, rr, ff)
            dc_prev = {'NRZ-L': A * (2 * p1 - 1), 'RZ': A * p1 / 2,
                       'RZ-polar': A * (2 * p1 - 1) / 2}.get(nome, 0.0)

            figura_dep(f, Pw, cont_i, raia_i, rot_i, cont_r, raias_r, dc, Rb, A, Tb,
                       f'{nome} - {ROTULO_FONTE[nome_fonte]}',
                       os.path.join(a.saida, f'dep_{nome}_{nome_fonte}.png'))

            linhas.append({
                'fonte': nome_fonte, 'codigo': nome, 'p1': p1,
                'dc_prev': dc_prev, 'dc_med': dc, 'potencia': pot,
                's_rb_db': rel_db, 'raia_rb_db': raia_db,
                'b90_teo': b90_teo / Rb, 'b90_rn': b90_rn / Rb, 'b90_med': b90 / Rb,
                'run_max': run_max, 'transicoes_bit': dens,
                'eps_max_pct': 100 * analise.erro_relogio_toleravel(run_max),
                'obs': extra,
            })

    figura_formas_de_onda(sps, A, pulsos, os.path.join(a.saida, 'formas_de_onda.png'))

    # Tabela de resultados
    cab = ('| Fonte | Código | P(bit=1) | DC previsto (V) | DC medido (V) | Potência (W) | '
           'S(Rb)/pico (dB) | Raia em Rb (dB) | B90 teórica ideal (×Rb) | B90 teórica com Rn (×Rb) | '
           'B90 medida (×Rb) | Run máx. (bits) | Transições/bit | Erro de relógio tolerável (%) |')
    sep = '|' + '---|' * 14
    texto = [cab, sep]
    for L in linhas:
        texto.append(
            f"| {L['fonte']} | {L['codigo']} | {L['p1']:.3f} | {L['dc_prev']:+.3f} | {L['dc_med']:+.3f} | "
            f"{L['potencia']:.3f} | {L['s_rb_db']:.1f} | {L['raia_rb_db']:.1f} | {L['b90_teo']:.2f} | "
            f"{L['b90_rn']:.2f} | {L['b90_med']:.2f} | {L['run_max']:.1f} | {L['transicoes_bit']:.2f} | "
            f"{L['eps_max_pct']:.3f} |")
    obs = [f"- {L['fonte']} / {L['codigo']}: {L['obs']}" for L in linhas if L['obs']]
    params = (f'Parâmetros: Rb = {Rb:g} bit/s, {sps} amostras/bit (fs = {fs:g} Hz), A = {A:g} V, '
              f'Welch com {a.segmento} bits/segmento (resolução {Rb / a.segmento:g} Hz), '
              f'imagem {a.tamanho}x{a.tamanho} modo {a.modo}, HDB3 com pulso {a.formato_hdb3.upper()}.')
    with open(os.path.join(a.saida, 'metricas.md'), 'w', encoding='utf-8') as fp:
        fp.write(params + '\n\n' + '\n'.join(texto) + '\n\n' + '\n'.join(obs) + '\n')
    print(params)
    print('\n'.join(texto))
    print('\n'.join(obs))
    print(f'\nFiguras e tabela salvas em ./{a.saida}/')


if __name__ == '__main__':
    main()