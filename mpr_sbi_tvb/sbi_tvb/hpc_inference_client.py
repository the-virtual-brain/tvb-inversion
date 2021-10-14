# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#
import os
import numpy as np
from tvb.config.init.datatypes_registry import populate_datatypes_registry
from tvb.core.neocom.h5 import load_ht


class HPCInferenceClient:
    def __init__(self, simulator_gid, input_dir, results_file, backend):
        populate_datatypes_registry()
        self.simulator_gid = simulator_gid
        self.input_dir = input_dir
        self.results_file = results_file
        self.backend = backend

    def run_simulation(self):
        loaded_sim = load_ht(self.simulator_gid, self.input_dir)
        loaded_sim.configure()
        (temporal_average_time, temporal_average_data), = self.backend().run_sim(loaded_sim,
                                                                                 simulation_length=loaded_sim.simulation_length)
        save_path = os.path.join(self.input_dir, self.results_file)
        np.savez(save_path, time=temporal_average_time, data=temporal_average_data)
