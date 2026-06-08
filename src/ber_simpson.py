"""
Estimacion de BER en sistema BPSK/AWGN
Metodo de Simpson 1/3 Compuesto
Grupo 2 - Metodos Numericos - PAO 4 - ESPOCH - 2026
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# =====================================================================
# BLOQUE 1: Parametros del sistema de comunicaciones
# =====================================================================
P_tx  = 1.0        # Potencia de transmision [W]
BW    = 1e6        # Ancho de banda [Hz]
R_b   = 1e6        # Tasa de bits [bps]
E_b   = P_tx / R_b # Energia por bit [J] = 1e-6 J
b_int = 8.0        # Limite superior de integracion
n     = 100        # Numero de subintervalos (debe ser par)

snr_dB  = np.arange(0, 10.5, 0.5)   # 21 puntos: 0, 0.5, ..., 10 dB
snr_lin = 10 ** (snr_dB / 10.0)     # conversion a escala lineal

# =====================================================================
# BLOQUE 2: Funcion integrando (densidad gaussiana estandar)
# =====================================================================
def integrando(t):
    """f(t) = exp(-t^2/2) / sqrt(2*pi)"""
    return np.exp(-0.5 * t**2) / np.sqrt(2.0 * np.pi)

# =====================================================================
# BLOQUE 3: Metodo de Simpson 1/3 compuesto
# =====================================================================
def simpson_compuesto(f, a, b, n):
    """
    Integra f sobre [a, b] con n subintervalos (n debe ser par).
    Retorna la aproximacion de la integral por Simpson 1/3 compuesto.
    """
    if n % 2 != 0:
        raise ValueError("n debe ser par para el Metodo de Simpson 1/3")
    if n <= 0:
        raise ValueError("n debe ser un entero positivo")
    h = (b - a) / n
    t = np.linspace(a, b, n + 1)  # n+1 nodos equidistantes
    y = f(t)
    coefs = np.ones(n + 1)
    coefs[1:-1:2] = 4   # indices impares: coeficiente 4
    coefs[2:-2:2] = 2   # indices pares internos: coeficiente 2
    return (h / 3.0) * np.dot(coefs, y)

# =====================================================================
# BLOQUE 4: Calculo de BER para todos los puntos de SNR
# =====================================================================
ber_simp = np.zeros(len(snr_dB))
ber_erfc = np.zeros(len(snr_dB))
err_rel  = np.zeros(len(snr_dB))

for idx, gamma in enumerate(snr_lin):
    x = np.sqrt(2.0 * gamma)  # argumento de Q(x)
    ber_simp[idx] = simpson_compuesto(integrando, x, b_int, n)
    ber_erfc[idx] = 0.5 * erfc(np.sqrt(gamma))
    err_rel[idx]  = (abs(ber_simp[idx] - ber_erfc[idx])
                     / ber_erfc[idx])

print(f"{'SNR(dB)':>8} | {'BER_Simpson':>14} | "
      f"{'BER_erfc':>14} | {'Err_Rel':>12}")
print("-" * 56)
for i in range(len(snr_dB)):
    print(f"{snr_dB[i]:8.1f} | {ber_simp[i]:14.6e} | "
          f"{ber_erfc[i]:14.6e} | {err_rel[i]:12.3e}")

# =====================================================================
# BLOQUE 5: Analisis de convergencia (SNR = 5 dB)
# =====================================================================
snr_test  = 5.0
gamma_t   = 10 ** (snr_test / 10.0)
x_test    = np.sqrt(2.0 * gamma_t)
ber_ref   = 0.5 * erfc(np.sqrt(gamma_t))

n_values  = [4, 8, 16, 32, 64, 128, 256, 512]
err_conv  = []
h_values  = []
prev_err  = None

print("\nAnalisis de convergencia para SNR = 5 dB:")
print(f"{'n':>5} | {'h':>10} | {'Err_Rel':>12} | {'Factor':>8}")
print("-" * 42)
for ni in n_values:
    hi    = (b_int - x_test) / ni
    ber_i = simpson_compuesto(integrando, x_test, b_int, ni)
    ei    = abs(ber_i - ber_ref) / ber_ref
    factor = (prev_err / ei) if prev_err else float('nan')
    h_values.append(hi)
    err_conv.append(ei)
    print(f"{ni:5d} | {hi:10.6f} | {ei:12.3e} | {factor:8.2f}")
    prev_err = ei

log_h = np.log10(h_values)
log_e = np.log10(err_conv)
coef  = np.polyfit(log_h[1:], log_e[1:], 1)
print(f"\nPendiente (orden de convergencia): {coef[0]:.4f}")

# =====================================================================
# BLOQUE 6: Graficas
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.semilogy(snr_dB, ber_simp, 'bo-',
             label='Simpson 1/3 (n=100)', markersize=5)
ax1.semilogy(snr_dB, ber_erfc, 'r--',
             label='Referencia erfc (SciPy)', linewidth=2)
ax1.axhline(1e-3, color='gray', linestyle=':',
            alpha=0.7, label='Umbral VoIP ($10^{-3}$)')
ax1.axhline(1e-6, color='purple', linestyle=':',
            alpha=0.7, label='Umbral datos ($10^{-6}$)')
ax1.set_xlabel('SNR (dB)')
ax1.set_ylabel('BER')
ax1.set_title('BER vs. SNR -- BPSK/AWGN')
ax1.legend(fontsize=9)
ax1.grid(True, which='both', alpha=0.4)

ax2 = axes[1]
h_ref = np.array(h_values)
ax2.loglog(h_values, err_conv, 'gs-',
           label='Error relativo Simpson', markersize=7)
ax2.loglog(h_ref,
           (err_conv[3] / h_values[3]**4) * h_ref**4,
           'k--', label='Pendiente 4 (ref.)', linewidth=1.5)
ax2.set_xlabel('Tamano de paso h')
ax2.set_ylabel('Error relativo')
ax2.set_title(f'Convergencia: pendiente aprox. {coef[0]:.3f}')
ax2.legend()
ax2.grid(True, which='both', alpha=0.4)

plt.tight_layout()
plt.savefig('Figura1_BER_vs_SNR.png', dpi=150, bbox_inches='tight')
plt.savefig('Figura2_Convergencia_Simpson.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n¡Graficas guardadas como 'Figura1_BER_vs_SNR.png' y 'Figura2_Convergencia_Simpson.png'!")