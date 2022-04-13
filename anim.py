import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random as rnd
from functools import reduce
import networkx as nx
from dataclasses import dataclass
from time import sleep
from matplotlib import rc
from sys import argv
from pathlib import Path
import multiprocessing as mp
import json

def bind_neighbours(w, h, ns):
  return [(x % w, y % h) for x, y in ns] #  if 0 <= p[0] < w and 0 <= p[1] < h
  
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

@dataclass
class Stepper:
  V_max: float
  V_min: float
  V_step: float

class Config:
  """
  Single MC config
  """

  size: int

  spins:   list[list[ int ]]
  lattice: list[list[ list[int] ]]
  energy:  float
  time:    int

  spin_total: float

  H:  float
  J:  float
  T:  float
  kb: float

  fig: plt.Figure
  axs: dict[str, plt.Axes]
  name: str

  # Data points. |Hs| = |Ms| = |Ts|
  Hs: list[float] # [H] list of 
  Ms: list[float] # [M]
  Ts: list[float] # [T]

  H_stepper: Stepper
  steps_per_H: int


  def __init__(self, size: int,  H: float, J: float, T: float, kb: float, probability=0.5, mosaic: list[list[str]]=[['lattice', 'spins']], make_network=make_network2, name='', H_step: float = 0.1, H_max: float = 1, steps_per_H: int = 1):
    self.spins, self.lattice = make_network(size, size, probability) # make_network2(size, size, probability) # make_network_regular(size, size)
    self.H = H
    self.J = J
    self.T = T
    self.kb = kb
    self.fig, self.axs = plt.subplot_mosaic(mosaic, figsize=(5.5, 3.5), constrained_layout=True)
    self.name = name

    self.size = size
    self.time = 0
    self.spin_total = sum([si for column in self.spins for si in column])
    self.energy = self.calc_energy()

    self.H_stepper = Stepper(H_max, -H_max, H_step)
    self.steps_per_H = steps_per_H

    self.Hs = []
    self.Ms = []
    self.Ts = []


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
    Hh   = -self.H * self.spin_total

    return Hint + Hh


  def calc_magnetisation(self) -> float:
    return self.spin_total / (self.size * self.size)


  ####################################
  # 		       Evolution             #
  ####################################

  def step_spin(self, position) -> list[tuple[int, int]]:
    x, y = position

    dH = self.calc_delta_H((x, y))

    if dH <= 0 or rnd.random() < np.exp(-dH/(self.kb * self.T)):
      return [(x, y)]
    return []


  def evolve_single_spin(self):
    for z in range(self.size*self.size):
      x = rnd.choice(range(self.size))
      y = rnd.choice(range(self.size))

      for xx, yy in self.step_spin((x, y)):
        self.spins[x][y] *= -1
        self.spin_total += 2 * self.spins[x][y]
        self.energy += self.calc_delta_H((xx,yy))

  ####################################
  # 		       Plotting              #
  ####################################

  def plot_lattice(self):
    for x in range(self.size):
      for y in range(self.size):
        for xx, yy in self.lattice[x][y]:
          self.axs['lattice'].plot([x, xx], [y, yy], 'k-')


  def plot_spins(self):
    self.fig.suptitle(f'{self.name}, t={self.time}, M={self.calc_magnetisation()}, H={self.H}')
    self.axs['spins'].imshow(self.spins, cmap='binary', vmin=-1, vmax=1)
    # plt.show(block=False)


  def plot_magnetisation(self):
    self.axs['magnetisation'].plot(self.Hs, self.Ms, 'c-')


  ####################################
  # 		    Data Gathering           #
  ####################################

  def gather_data(self):
    self.Ts.append(self.T)
    self.Ms.append(self.calc_magnetisation())
    self.Hs.append(self.H)
  
  def save_data(self, dir_base: str):
    plot_path = Path(dir_base) / 'plots' / f'name_{self.name},size_{self.size},T_{self.T},time_{self.time}.png'
    data_path = Path(dir_base) / 'data' / f'name_{self.name}.json'

    with open(data_path, 'w+') as f:
      f.write(self.toJSON())
    
    f = open(plot_path, 'w+')
    f.close()
    self.fig.savefig(plot_path)
      

  def toJSON(self) -> str:
    return json.dumps(dict(
      name=self.name,
      size=self.size,
      time=self.time,
      Ms=self.Ms,
      Ts=self.Ts,
      Hs=self.Hs,
      spins=self.spins,
      lattice=self.lattice,
      M=self.calc_magnetisation(),
      E=self.calc_energy(),
      J=self.J,
      T=self.T,
      kb=self.kb,
      H_step=self.H_stepper.V_step,
      H_max=self.H_stepper.V_max,
      H_min=self.H_stepper.V_min,
      steps_per_H=self.steps_per_H
    ), sort_keys=True, separators=(',', ':'))

  def simulate_step(self, director):
    director(self)
    self.time += 1

    self.evolve_single_spin()


def make_director():
  def director(c: Config):
    dirname = (argv[1] if len(argv) > 1 else 'out')
    figname = f'{dirname}/{c.name}__size_{c.size}__T_{c.T}__time_{c.time}.png'

    if c.H <= c.H_stepper.V_min:
      c.H_stepper.V_step = np.abs(c.H_stepper.V_step)
    elif c.H >= c.H_stepper.V_max:
      c.H_stepper.V_step = -np.abs(c.H_stepper.V_step)

    if c.time % c.steps_per_H == 0:
      print(f'H: {c.H}, step: {c.H_stepper.V_step}')
      c.gather_data()
      c.plot_spins()
      c.plot_magnetisation()
      c.save_data(dirname)
      plt.pause(0.00001)
      # c.fig.savefig(figname)
      c.H = round(c.H + c.H_stepper.V_step, 10)

  return director

def main(size: int, H: float, J: float, T: float, kb: float, probability: float, mosaic: list[list[str]], name: str, H_max: float, H_step: float, make_network, steps_per_H: int):
  reached_max = False

  c = Config(
    size=size, H=H, J=J, T=T, kb=kb, probability=probability, mosaic=mosaic, 
    make_network=make_network, 
    name=name, 
    H_max=H_max, H_step=H_step, steps_per_H=steps_per_H
    )

  plt.ioff()

  while not (reached_max and c.H_stepper.V_step > 0):
    reached_max = c.H == H_max
    c.simulate_step(make_director())




if __name__ == '__main__':
  sleep(2)
  size        = 100
  H           = 0.00
  J           = 1
  T           = 0.5
  kb          = 1
  H_max       = 0.3
  steps_per_H = 100
  H_step      = H_max/(2*steps_per_H)
  probability = 0.5
  mosaic      = [['spins'], ['magnetisation']]

  with mp.Pool() as pool:
    pool.starmap(main, 
      [
        (size, H, J, T, kb, probability, mosaic, _name, H_max, H_step, _make_network, steps_per_H)
        for _name, _make_network in [('irregular', make_network2), ('regular', make_network_regular)]
      ])

  """
  Cir = Config(
    size=d, H=H, J=J, T=T, kb=kb, probability=probability, mosaic=mosaic.copy(), 
    make_network=make_network2, 
    name='irregular', 
    H_max=H_max, H_step=0.01
    )

  Cr = Config(
    size=d, H=H, J=J, T=T, kb=kb, probability=probability, mosaic=mosaic.copy(), 
    make_network=make_network_regular, 
    name='regular', 
    H_max=H_max, H_step=0.01
    )
  
  plt.ion()

  while True:      
    Cir.simulate_step(director)
    Cr.simulate_step(director)
  """
