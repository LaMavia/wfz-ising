import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random as rnd
from functools import reduce
import networkx as nx
from dataclasses import dataclass
from time import sleep
from matplotlib import rc

def bind_neighbours(w, h, ns):
  return [p for p in ns if 0 <= p[0] < w and 0 <= p[1] < h]
  
def get_potential_neighbours_even(d, p) -> list[int]:
  w, h = d
  x, y = p

  raw = [(x + dx, y + dy) 
          for dx in [-1, 0, 1] 
          for dy in [-1, 0, 1] 
          if not (dx == dy == 0)
        ]

  return bind_neighbours(w, h, raw)

def get_potential_neighbours_odd(d, p) -> list[int]:
  w, h = d
  x, y = p

  return bind_neighbours(w, h, [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])

def get_potential_neighbours_even_odd(d, p) -> list[int]:
  w, h = d
  x, y = p

  return bind_neighbours(w, h, [(x - 1, y - 1), (x - 1, y + 1), (x + 1, y - 1), (x + 1, y + 1)])

def make_network2(w: int, h: int, p: float):
  rnd.seed(1)

  def is_connected() -> bool:
    # return True
    return rnd.random() >= p
  
  ns = [[[] for y in range(0, h)] for x in range(0, w)]
  s = [[rnd.choice([-1, 1]) for y in range(0, h)] for x in range(0, w)]

  for y in range(0, h):
    for x in range(0, w):
      my = y % 2
      mx = x % 2

      f = lambda d, p: []
      if mx == my == 0:
        f = get_potential_neighbours_even
      elif mx == my == 1:
        f = get_potential_neighbours_odd
      elif mx == 0 and my == 1:
        f = get_potential_neighbours_even_odd

      pn = [e for e in f((w, h), (x, y)) if is_connected()] 

      ns[x][y].extend(pn)

      for xx, yy in pn:
        ns[xx][yy].append((x,y))

  return s, ns

def make_network_regular(w: int, h: int, *_):
  rnd.seed(1)

  ns = [[[] for y in range(0, h)] for x in range(0, w)]
  s = [[rnd.choice([-1, 1]) for y in range(0, h)] for x in range(0, w)]

  for y in range(h):
    for x in range(w):
      ns[x][y].extend(bind_neighbours(w, h, [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]))

  return s, ns

class Config:
  """
  Single MC config
  """

  spins:   list[list[ int ]]
  lattice: list[list[ list[int] ]]
  energy: float

  H: float
  J: float
  T: float
  kb: float

  fig: plt.Figure
  axs: dict[str | int, plt.Axes]
  name: str

  def __init__(self, size: int,  H: float, J: float, T: float, kb: float, probability=0.5, mosaic: list[list[str]]=[['lattice', 'spins']], make_network=make_network2, name=''):
    self.spins, self.lattice = make_network(size, size, probability) # make_network2(size, size, probability) # make_network_regular(size, size)
    self.H = H
    self.J = J
    self.T = T
    self.kb = kb
    self.fig, self.axs = plt.subplot_mosaic(mosaic, figsize=(5.5, 3.5), constrained_layout=True)
    self.name = name

    self.energy = self.calc_energy()

  """
  \Delta \mathcal{H} \{s_k -> -s_k\} = s_k (J \sum_{j \in \Delta_k} s_j + 2H)
  """
  def calc_delta_H(self, position: tuple[int, int]) -> float:
    x, y = position
    sk = self.spins[x][y]
    neighbours = [self.spins[xx][yy] for xx, yy in self.lattice[x][y]]

    return sk * ( self.J * sum(neighbours) + 2*self.H )

  def calc_energy(self) -> float:
    Hint = -1/2 * self.J * sum([ si * sum([self.spins[xx][yy] for xx, yy in self.lattice[x][y]]) for x, column in enumerate(self.spins) for y, si in enumerate(column) ])
    Hh   = -self.H * sum([si for column in self.spins for si in column])

    return Hint + Hh

  def calc_magnetisation(self) -> float:
    return sum([ si for column in self.spins for si in column ]) / sum([ len(column) for column in self.spins ])

  def step_spin(self, position) -> list[tuple[int, int]]:
    x, y = position

    dH = self.calc_delta_H((x, y))

    if dH <= 0 or rnd.random() < np.exp(-dH/(self.kb * self.T)):
      return [(x, y)]
    return []

  def evolve_single_spin(self):
    for z in range(len(self.spins)**2):
      x = rnd.choice(range(len(self.spins)))
      y = rnd.choice(range(len(self.spins[0])))

      for xx, yy in self.step_spin((x, y)):
        self.spins[x][y] *= -1
        self.energy += self.calc_delta_H((xx,yy))


  def evolve_all_spins(self):
    for x, column in enumerate(self.spins):
      for y, sk in enumerate(column):
        for xx, yy in self.step_spin((x, y)):
          self.spins[x][y] *= -1
          self.energy += self.calc_delta_H((xx,yy))


  def plot_lattice(self):
    h = len(self.spins)
    w = len(self.spins[0]) 

    for x, column in enumerate(self.lattice):
      for y, ns in enumerate(column):
        for xx, yy in ns:
          self.axs['lattice'].plot([x, xx], [y, yy], 'k-')

  def plot_spins(self, time):
    self.fig.suptitle(f'{self.name}, t={time}, M={self.calc_magnetisation()}, H={self.H}')
    self.axs['spins'].imshow(self.spins, cmap='binary', vmin=-1, vmax=1)
    # plt.show(block=False)

if __name__ == '__main__':
  sleep(2)
  d = 500
  t = 0
  H = 0.00
  J = 1
  T = 0.5
  kb = 1
  probability = 0.5
  mosaic = [['spins']]

  Cir = Config(size=d, H=H, J=J, T=T, kb=kb, probability=probability, mosaic=mosaic, make_network=make_network2, name='irregular')
  Cr  = Config(size=d, H=H, J=J, T=T, kb=kb, probability=probability, mosaic=mosaic, make_network=make_network_regular, name='regular')
  
  plt.ion()
  

  while True:
    t += 1

    Cir.evolve_single_spin()
    Cr.evolve_single_spin()

    Cir.plot_spins(time=t)
    Cr.plot_spins(time=t)
    # im.set_array(c.spins)

    plt.pause(0.0001)
    # sleep(0.01)

  plt.close()
