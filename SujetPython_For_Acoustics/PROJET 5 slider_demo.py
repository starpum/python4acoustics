"""
Slider and button Demo
======================
 Lisez bien les commentaires !
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

fig, ax = plt.subplots()  # création du cadre qui contiendra le sinus
plt.subplots_adjust(bottom=0.25) # position de la ligne inférieure du cadre
								 # 0=bas de la figure, 1=haut de la figure

# création du sinus
t = np.arange(0.0, 1.0, 0.001)
a0 = 5
f0 = 3
s = a0*np.sin(2*np.pi*f0*t)

# Tracé de la courbe et enregistrement dans "ligne" pour pouvoir la modifier
ligne, = plt.plot(t, s, lw=2, color='red')
ax.set_title('Sinus à la fréquence {:6.2f} Hz'.format(f0))
plt.axis([0, 1, -10, 10])

# création d'un cadre qui contiendra la glissière.
# les coordonnées [position coin inférieur gauche, longueur, hauteur]
# de la glissière avec position (0,0)= en bas à droite
# position (1,1)=en haut à droite de la figure
# longueur = 1 (100% de la largeur de la figure)
# hauteur = 1 (100% de la hauteur de la figure)
axfreq = plt.axes([0.15, 0.1, 0.75, 0.05], facecolor='lightgoldenrodyellow')
# création de la glissière dans ce cadre
sfreq = Slider(axfreq, 'Fréquence', 0.1, 30.0, valinit=f0)

# fonction qui sera appelée lorsqu'on déplace la glissière
def miseAjour(val):
    freq = sfreq.val  		# on récupère la valeur de la glissière
    ligne.set_ydata(a0*np.sin(2*np.pi*freq*t)) # on met à jour la ligne
    ax.set_title('Sinus à la fréquence {:6.2f} Hz'.format(freq)) # le titre

# on attribue la fonction précédente
# à une modification de la valeur de la glissière
sfreq.on_changed(miseAjour)

# création d'un cadre qui contiendra le bouton 'Remise à zéro'
resetax = plt.axes([0.7, 0.025, 0.20, 0.04])
# création du bouton dans ce cadre
bouton = Button(resetax, 'Remise à zéro', color='red', hovercolor='lightblue')

def miseAzero(event):
    sfreq.reset()  # réinitialisation de la glissière

# on attribue la fonction précédente à un clic sur le bouton
bouton.on_clicked(miseAzero)

plt.show()