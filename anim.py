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

def make_network_regular(w: int, h: int):
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
  H: float
  J: float
  T: float
  kb: float

  def __init__(self, size: int,  H: float, J: float, T: float, kb: float, probability=0.5):
    self.spins, self.lattice = make_network2(size, size, probability) # make_network_regular(size, size)
    self.H = H
    self.J = J
    self.T = T
    self.kb = kb

  """
  \Delta \mathcal{H} \{s_k -> -s_k\} = s_k (J \sum_{j \in \Delta_k} s_j + 2H)
  """
  def calc_delta_H(self, position: tuple[int, int]) -> float:
    x, y = position
    sk = self.spins[x][y]
    neighbours = [self.spins[xx][yy] for xx, yy in self.lattice[x][y]]

    return sk * ( self.J * sum(neighbours) + 2*self.H )

  def calc_energy(self) -> float:
    Hint = -1/2 * self.J * sum([ si * sum(self.lattice[x][y]) for x, column in enumerate(self.spins) for y, si in enumerate(column) ])
    Hh   = -self.H * sum([si for column in self.spins for si in column])

    return Hint + Hh

  def calc_magnetisation(self) -> float:
    return sum([ si for column in self.spins for si in column ]) / sum([ len(column) for column in self.spins ])

  def evolve_single_spin(self):
    spins_to_switch = []

    # for x, column in enumerate(self.spins):
    #   for y, sk in enumerate(column):
    #     if (rnd.random() < 0.5): continue

    for z in range(len(self.spins)):
      x = rnd.choice(range(len(self.spins)))
      y = rnd.choice(range(len(self.spins[0])))

      dH = self.calc_delta_H((x, y))

      if dH <= 0 or rnd.random() < np.exp(-dH/(self.kb * self.T)):
        spins_to_switch.append((x, y))

    for x, y in spins_to_switch:
      self.spins[x][y] *= -1
      
  def plot_lattice(self, dpi=100):
    h = len(self.spins)
    w = len(self.spins[0])

    fig = plt.figure(figsize=(2,2), dpi=dpi)
    ax = fig.add_subplot(111)

    fig.tight_layout()

    for x, column in enumerate(self.lattice):
      for y, ns in enumerate(column):
        for xx, yy in ns:
          ax.plot([x, xx], [y, yy], 'k-')

if __name__ == '__main__':
  d = 100
  t = 0
  c = Config(size=d, H=0.1, J=1, T=0.5, kb=1, probability=0.5)

  # plot_graph(d, c.lattice, 100)
  c.plot_lattice()

  fig, ax = plt.subplots()
  im = ax.imshow(c.spins, 'binary')
  

  while True:
    t += 1

    c.evolve_single_spin()

    fig.suptitle(f't={t}, M={c.calc_magnetisation()}, H={c.H}')
    im.set_array(c.spins)

    plt.pause(0.001)
    sleep(0.01)

  plt.close()
