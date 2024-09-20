import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Générer les données pour ctauLLP1
ctauLLP1 = np.linspace(0, 10, 100)
cross_section1 = np.exp(-0.1 * ctauLLP1)  # Exemple de fonction pour cross section

# Générer les données pour ctauLLP2
ctauLLP2 = np.linspace(0, 10, 100)
cross_section2 = np.exp(-0.2 * ctauLLP2)  # Exemple de fonction pour cross section

# Créer le graphique en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Tracer les données de ctauLLP1 en fonction de cross section
ax.plot(ctauLLP1, np.zeros_like(ctauLLP1), cross_section1, label=r'LLP1')

# Tracer les données de ctauLLP2 en fonction de cross section
ax.plot(np.zeros_like(ctauLLP2), ctauLLP2, cross_section2, label=r'LLP2')

# Nommer les axes
ax.set_xlabel(r'c$\tau$LLP1 [m]', fontsize=13)
ax.set_ylabel(r'c$\tau$LLP2 [m]', fontsize=13)
ax.set_zlabel(r'95% CL limit on $\sigma \times B$ [pb]', fontsize=13 )


# Ajouter une légende
ax.legend()

# Afficher le graphique
plt.savefig("Test3D.png")
plt.close()