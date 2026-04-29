import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


archivo_sin_masa = Path("datostromposinmasa.txt")
archivo_con_masa = Path("datostrompomasa.txt")


def leer_datos_tracker(ruta):
    """
    Lee datos exportados desde Tracker.
    Sirve aunque el archivo use coma decimal, tabulaciones o comas.
    Devuelve columnas:
    t          tiempo en segundos
    theta_deg  ángulo acumulado en grados
    """

    tiempos = []
    angulos = []

    with open(ruta, "r", encoding="utf-8-sig") as archivo:
        for linea in archivo:
            linea = linea.strip()

            if not linea:
                continue

            numeros = re.findall(r"[-+]?\d+(?:[,.]\d+)?", linea)

            if len(numeros) >= 2:
                t = float(numeros[0].replace(",", "."))
                theta = float(numeros[1].replace(",", "."))

                tiempos.append(t)
                angulos.append(theta)

    return pd.DataFrame({
        "t": tiempos,
        "theta_deg": angulos
    })


datos_sin = leer_datos_tracker(archivo_sin_masa)
datos_con = leer_datos_tracker(archivo_con_masa)

n_sin_original = len(datos_sin)
n_con_original = len(datos_con)

N = min(n_sin_original, n_con_original)

datos_sin = datos_sin.iloc[:N].reset_index(drop=True)
datos_con = datos_con.iloc[:N].reset_index(drop=True)

print(f"Datos originales sin masa: {n_sin_original}")
print(f"Datos originales con masa: {n_con_original}")
print(f"Datos usados en cada caso: {N}")


# Convertir ángulo acumulado de grados a radianes
datos_sin["theta_rad"] = np.deg2rad(datos_sin["theta_deg"])
datos_con["theta_rad"] = np.deg2rad(datos_con["theta_deg"])


# Calcular velocidad angular
# np.gradient conserva la misma cantidad de datos.
datos_sin["omega_rad_s"] = np.gradient(datos_sin["theta_rad"], datos_sin["t"])
datos_con["omega_rad_s"] = np.gradient(datos_con["theta_rad"], datos_con["t"])

# También en grados/s, por si quieres comparar directamente con los datos originales
datos_sin["omega_deg_s"] = np.gradient(datos_sin["theta_deg"], datos_sin["t"])
datos_con["omega_deg_s"] = np.gradient(datos_con["theta_deg"], datos_con["t"])


resultados = pd.DataFrame({
    "t_sin_masa": datos_sin["t"],
    "theta_sin_masa_deg": datos_sin["theta_deg"],
    "theta_sin_masa_rad": datos_sin["theta_rad"],
    "omega_sin_masa_rad_s": datos_sin["omega_rad_s"],
    "omega_sin_masa_deg_s": datos_sin["omega_deg_s"],

    "t_con_masa": datos_con["t"],
    "theta_con_masa_deg": datos_con["theta_deg"],
    "theta_con_masa_rad": datos_con["theta_rad"],
    "omega_con_masa_rad_s": datos_con["omega_rad_s"],
    "omega_con_masa_deg_s": datos_con["omega_deg_s"]
})

resultados.to_csv("velocidades_angulares_trompo.csv", index=False)

print(resultados.head())

print("\nResumen velocidad angular sin masa [rad/s]:")
print(datos_sin["omega_rad_s"].describe())

print("\nResumen velocidad angular con masa [rad/s]:")
print(datos_con["omega_rad_s"].describe())


plt.figure(figsize=(9, 5))
plt.plot(datos_sin["t"], datos_sin["theta_deg"], marker="o", label="Sin masa extra")
plt.plot(datos_con["t"], datos_con["theta_deg"], marker="s", label="Con masa extra")
plt.xlabel("Tiempo t [s]")
plt.ylabel("Ángulo acumulado θ [grados]")
plt.title("Ángulo acumulado del trompo en función del tiempo")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("angulo_vs_tiempo_trompo.png", dpi=300)
plt.show()


plt.figure(figsize=(9, 5))
plt.plot(datos_sin["t"], datos_sin["omega_rad_s"], marker="o", label="Sin masa extra")
plt.plot(datos_con["t"], datos_con["omega_rad_s"], marker="s", label="Con masa extra")
plt.xlabel("Tiempo t [s]")
plt.ylabel("Velocidad angular ω [rad/s]")
plt.title("Velocidad angular del trompo en función del tiempo")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("velocidad_angular_trompo.png", dpi=300)
plt.show()