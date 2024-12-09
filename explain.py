'''
Explanatory models in Neonatology (EXPLAIN)
----------------------------------------------------------------------------------------------------------------------------
This python script contains the complete documented source code of explain which is an integrated model of human physiology
The main purpose of this python script is to stay as close to the scientific papers and equations as possible.
The code is therefore designed for readability and not optimized for performance.

The explain model benefits a lot (>10x speed) from the JIT-compiler of the PyPy3 implementation of Python3 (https://pypy.org).
The best way to run the model is by using an interactive python environment like a jupyter notebook/lab of any other.
We recommend the interactive python notebooks in VS Code with PyPy3 as the engine.

Don't forget to install the three dependencies of explain: matplotlib, numpy and ipykernel (if running in an interactive notebook)
See at the bottom of this file for instructions on how to run this model and on how to install PyPy3 on Windows/Linux/MacOS

Go to https://explain-user.com for the complete integrated model of the neonate running in realtime as a webapplication with an
user-friendly graphical user interface. This webapplication is programmed in JS and Webassembly and is much faster and capable 
of running the models in realtime. However, you don't have access to the source code.

If you have problems running this code or other questions please contact Tim Antonius at tim.antonius@radboudumc.nl

Have fun!
'''
#----------------------------------------------------------------------------------------------------------------------------
# import the dependencies. 
import math, random, json                               # general modules used by most objects  
from time import perf_counter                           # used to generate model performance parameters
import matplotlib.pyplot as plt                         # used by the plotter object
import numpy as np                                      # used by the plotter object

#----------------------------------------------------------------------------------------------------------------------------
# explain core models
class BaseModelClass():
    # This base model class is the blueprint for all the model objects (classes).
    # It incorporates the properties and methods which all model objects implement
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize independent properties which all models implement
        self.name: str = name                           # name of the model object
        self.description: str = ""                      # description in for documentation purposes
        self.is_enabled: bool = False                   # flag whether the model is enabled or not
        self.model_type: str = ""                       # holds the model type e.g. BloodCapacitance

        # initialize local properties
        self._model_engine: object = model_ref          # object holding a reference to the model engine
        self._t: float = model_ref.modeling_stepsize    # setting the modeling stepsize
        self._is_initialized: bool = False              # flag whether the model is initialized or not

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model as provided by args dictionary
        for key, value in args.items():
            setattr(self, key, value)
        
        # flag that the model is initialized
        self._is_initialized = True

    def step_model(self) -> None:
        # this method is called by the model engine and if the model is enabled and initialized it will do the model calculations
        if self.is_enabled and self._is_initialized:
            self.calc_model()
    
    def calc_model(self) -> None:
        # this method is overriden by almost all model classes as this is the place where model calculations take place
        pass

class Blood(BaseModelClass):
    '''
    The Blood model takes care of the blood composition. 
    It sets the solutes and gas concentrations on all blood containing models when explain starts. 
    It also houses the routines for acidbase and oxygenation calculations.

    Reference for acidbase calculations:
    Antonius TAJ, van Meurs WWL, Westerhof BE, de Boode WP. 
    A white-box model for real-time simulation of acid-base balance in blood plasma. 
    Adv Simul (Lond). 2023 Jun 15;8(1):16. doi: 10.1186/s41077-023-00255-2. PMID: 37322544; PMCID: PMC10268443.

    Reference for oxygen dissociation curve calculations:
    Siggaard-Andersen O, Wimberley PD, Göthgen I, Siggaard-Andersen M. 
    A mathematical model of the hemoglobin-oxygen dissociation curve of human blood and of the oxygen partial pressure as a function of temperature. 
    Clin Chem. 1984 Oct;30(10):1646-51. PMID: 6478594.

    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties
        self.viscosity: float = 6.0                     # blood viscosity (centiPoise = Pa *s)
        self.temp: float = 37.0                         # temperature (dgs C)
        self.to2: float = 0.0                           # total oxygen concentration (mmol/l)
        self.tco2: float = 0.0                          # total carbon dioxide concentration (mmol/l)
        self.solutes: dict = {}                         # dictionary holding the initial circulating solutes

        # -----------------------------------------------
        # initialize dependent properties
        self.preductal_art_bloodgas:object = {}         # dictionary containing the preductal arterial bloodgas
        self.art_bloodgas: object = {}                  # dictionary containing the (postductal) arterial bloodgas
        self.ven_bloodgas: object = {}                  # dictionary containing the venous bloodgas
        self.art_solutes: object = {}                   # dictionary containing the arterial solute concentrations

        # -----------------------------------------------
        # initialize local properties (preceded with _)
        self._blood_containing_modeltypes: list = ['BloodCapacitance', 'BloodTimeVaryingElastance', 'BloodPump']
        self._update_interval: float = 1.0              # interval at which the calculations are done
        self._update_counter: float = 0.0               # update counter intermediate
        self._ascending_aorta = None                    # object holding a reference to the ascending aorta model
        self._descending_aorta = None                   # object holding a reference to the descending aorta model
        self._right_atrium = None                       # object holding a reference to the right atrium

        # blood composition (acidbase and oxygenation) constants
        self._brent_accuracy: float = 1e-6              # accuracy of the brent root finding procedure
        self._max_iterations: float = 100               # maximum of iterations allowd by the brent root finding procedure
        self._kw: float = 0.000000000025119             # composite dissociation constant for water (mmol/l)
        self._kc: float = 0.000794328235                # dissociation constant for carbonic acid (mmol/l)
        self._kd: float = 0.000000060255959             # dissociation constant for bicarbonate ions (mmol/l)
        self._alpha_co2p: float = 0.03067               # carbon dioxide solubility coefficient (mmol/l*mmHg)
        self._left_hp: float = 0.000005848931925        # root finding procedure for acidbase [H+] left limit
        self._right_hp: float = 0.000316227766017       # root finding procedure for acidbase [H+] right limit
        self._left_o2: float = 0.01                     # root finding procedure for oxygenation po2 left limit
        self._right_o2: float = 800.0                   # root finding procedure for oxygenation po2 right limit
        self._gas_constant: float = 62.36367            # ideal gas law gas constant (L·mmHg/(mol·K))
        self._dpg: float = 5.0                          # 2,3-diphosphoglycerate concentration (mmol/l)

        # acid base and oxygenation intermediates used by the acidbase and oxygenation calculations
        self._tco2: float = 0.0
        self._sid: float = 0.0
        self._albumin: float = 0.0
        self._phosphates: float = 0.0
        self._uma: float = 0.0
        self._ph: float = 0.0
        self._pco2: float = 0.0
        self._hco3: float = 0.0
        self._be: float = 0.0
        self._to2: float = 0.0
        self._temp: float = 0.0
        self._po2: float = 0.0
        self._so2: float = 0.0

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)
        
        # set the solutes and temperature of the blood containing components
        for _, model in self._model_engine.models.items():
            if model.model_type in self._blood_containing_modeltypes:
                    model.to2 = self.to2
                    model.tco2 = self.tco2
                    model.solutes = {**self.solutes}
                    model.temp = self.temp
                    model.viscosity = self.viscosity

        # get the components where we measure the bloodgases
        self._ascending_aorta = self._model_engine.models["AA"]
        self._descending_aorta = self._model_engine.models["AD"]
        self._right_atrium = self._model_engine.models["RA"]

        # copy the initial arterial solutes
        self.art_solutes = {**self.solutes}

        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self) -> None:
        # for performance reasons the acidbase and oxygenation of the arterial and venous system 
        # is not calculated every model stop but every second
        self._update_counter += self._t
        if self._update_counter >= self._update_interval:
            self._update_counter = 0.0
           
            # preductal arterial bloodgas
            self.calc_blood_composition(self._ascending_aorta)
            self.preductal_art_bloodgas = {
                "ph": self._ascending_aorta.ph,
                "pco2": self._ascending_aorta.pco2,
                "po2": self._ascending_aorta.po2,
                "hco3": self._ascending_aorta.hco3,
                "be": self._ascending_aorta.be,
                "so2": self._ascending_aorta.so2
            }

            # postductal arterial bloodgas
            self.calc_blood_composition(self._descending_aorta)
            self.art_bloodgas = {
                "ph": self._descending_aorta.ph,
                "pco2": self._descending_aorta.pco2,
                "po2": self._descending_aorta.po2,
                "hco3": self._descending_aorta.hco3,
                "be": self._descending_aorta.be,
                "so2": self._descending_aorta.so2
            }

            # venous bloodgas
            self.calc_blood_composition(self._right_atrium)
            self.ven_bloodgas = {
                "ph": self._right_atrium.ph,
                "pco2": self._right_atrium.pco2,
                "po2": self._right_atrium.po2,
                "hco3": self._right_atrium.hco3,
                "be": self._right_atrium.be,
                "so2": self._right_atrium.so2
            }

            # arterial solute concentrations
            self.art_solutes = {**self._descending_aorta.solutes}

    def set_temperature(self, new_temp: float) -> None:
        # store the new temperature
        self.temp = new_temp

        # apply the new temperature to do the blood containing compartments
        for _, model in self._model_engine.models.items():
            if model.model_type in self._blood_containing_modeltypes:
                    model.temp = new_temp

    def set_viscosity(self, new_viscosity: float) -> None:
        # store the new viscosity
        self.viscosity = new_viscosity
        
        # apply the new viscosity to all the blood containing models
        for _, model in self._model_engine.models.items():
            if model.model_type in self._blood_containing_modeltypes:
                    model.viscosity = new_viscosity

    def set_to2(self, new_to2: float, bc_site: str = "") -> None:
        # bc_site contains the blood containg model where you want the to2 to be set
        # if bc_site is not set then the to2 will be set on all blood containing models
        if bc_site:
            # set solute on a specific blood containg model
            self._model_engine.models[bc_site].to2 = new_to2
        else:
            # set solute on all blood containing models
            for _, model in self._model_engine.models.items():
                if model.model_type in self._blood_containing_modeltypes:
                        model.to2 = new_to2

    def set_tco2(self, new_tco2: float, bc_site: str = "") -> None:
        # bc_site contains the blood containg model where you want the tco2 to be set
        # if bc_site is not set then the tco2 will be set on all blood containing models
        if bc_site:
            # set solute on a specific blood containg model
            self._model_engine.models[bc_site].tco2 = new_tco2
        else:
            # set solute on all blood containing models
            for _, model in self._model_engine.models.items():
                if model.model_type in self._blood_containing_modeltypes:
                        model.tco2 = new_tco2

    def set_solute(self, solute: str, solute_value: float, bc_site: str = "") -> None:
        # bc_site contains the blood containg model where you want the solute to be set
        # if bc_site is not set then the solute will be set on all blood containing models
        if bc_site:
            # set solute on a specific blood containg model
            self._model_engine.models[bc_site].solutes[solute] = solute_value
        else:
            # set solute on all blood containing models
            for _, model in self._model_engine.models.items():
                if model.model_type in self._blood_containing_modeltypes:
                        model.solutes = {**self.solutes}

    def calc_blood_composition(self, bc: object) -> None:
        # get a reference to the solutes object of the component referenced in the bc object
        sol = bc.solutes

        # get the independent parameters for the acidbase routine
        self._tco2 = bc.tco2
        self._to2 = bc.to2
        self._sid = sol["na"] + sol["k"] + 2 * sol["ca"] + 2 * sol["mg"] - sol["cl"] - sol["lact"]
        self._albumin = sol["albumin"]
        self._phosphates = sol["phosphates"]
        self._uma = sol["uma"]
        self._hemoglobin = sol["hemoglobin"]
        self._temp = bc.temp

        # now try to find the hydrogen concentration at the point where the net charge of the plasma 
        # is zero within limits of the brent accuracy
        hp = self._brent_root_finding(self._net_charge_plasma, self._left_hp, self._right_hp, self._max_iterations, self._brent_accuracy)

        # if the result is valid then store it inside the component
        if hp > 0:
            self._be = (self._hco3 - 25.1 + (2.3 * self._hemoglobin + 7.7) * (self._ph - 7.4)) * (1.0 - 0.023 * self._hemoglobin)
            bc.ph = self._ph
            bc.pco2 = self._pco2
            bc.hco3 = self._hco3
            bc.be = self._be

        # now try to find a po2 estimate for which the difference between the po2 and 
        # the calculated to2 (from the po2 estimate) are within the limits of the brent accuracy
        po2 = self._brent_root_finding(self._oxygen_content, self._left_o2, self._right_o2, self._max_iterations, self._brent_accuracy)

        # if the result is valid then store it inside the component
        if po2 > -1:
            bc.po2 = self._po2
            bc.so2 = self._so2 * 100.0

    def _net_charge_plasma(self, hp_estimate: float) -> float:

        # Calculate the pH based on the current hp estimate
        self._ph = -math.log10(hp_estimate / 1000.0)

        # Calculate the plasma co2 concentration based on the total co2 in the plasma, hydrogen concentration, and the constants Kc and Kd
        cco2p = self._tco2 / (1.0 + self._kc / hp_estimate + (self._kc * self._kd) / math.pow(hp_estimate, 2.0))

        # Calculate the plasma hco3(-) concentration (bicarbonate)
        self._hco3 = (self._kc * cco2p) / hp_estimate

        # Calculate the plasma co3(2-) concentration (carbonate)
        co3p = (self._kd * self._hco3) / hp_estimate

        # Calculate the plasma OH(-) concentration (water dissociation)
        ohp = self._kw / hp_estimate

        # Calculate the pco2 of the plasma
        self._pco2 = cco2p / self._alpha_co2p

        # Calculate the weak acids (albumin and phosphates)
        a_base = self._albumin * (0.123 * self._ph - 0.631) + self._phosphates * (0.309 * self._ph - 0.469)

        # Calculate the net charge of the plasma
        netcharge = hp_estimate + self._sid - self._hco3 - 2.0 * co3p - ohp - a_base - self._uma

        # Return the net charge
        return netcharge

    def _oxygen_content(self, po2_estimate: float) -> float:
        # calculate the saturation from the current po2 from the current po2 estimate
        self._so2 = self._oxygen_dissociation_curve(po2_estimate)

        # calculate the to2 from the current po2 estimate
        # INPUTS: po2 in mmHg, so2 in fraction, hemoglobin in mmol/l
        # convert the hemoglobin unit from mmol/l to g/dL  (/ 0.6206)
        # convert to output from ml O2/dL blood to ml O2/l blood (* 10.0)
        to2_new_estimate = (0.0031 * po2_estimate + 1.36 * (self._hemoglobin / 0.6206) * self._so2) * 10.0

        # conversion factor for converting ml O2/l to mmol/l
        mmol_to_ml = (self._gas_constant * (273.15 + self._temp)) / 760.0

        # convert the ml O2/l to mmol/l
        to2_new_estimate = to2_new_estimate / mmol_to_ml

        # store the current estimate
        self._po2 = po2_estimate

        # calculate the difference between the real to2 and the to2 based on the new po2 estimate and return it to the brent root finding function
        return self._to2 - to2_new_estimate

    def _oxygen_dissociation_curve(self, po2_estimate: float) -> float:
        # calculate the saturation from the po2 depending on the ph, be, temperature and _dpg level.
        a = 1.04 * (7.4 - self._ph) + 0.005 * self._be + 0.07 * (self._dpg - 5.0)
        b = 0.055 * (self._temp + 273.15 - 310.15)
        x0 = 1.875 + a + b
        h0 = 3.5 + a
        x = math.log((po2_estimate * 0.1333));  # po2 in kPa
        y = x - x0 + h0 * math.tanh(0.5343 * (x - x0)) + 1.875

        # return the o2 saturation in fraction
        return 1.0 / (math.exp(-y) + 1.0)

    def _brent_root_finding(self, f: object, x0: float, x1: float, max_iter: float, tolerance: float) -> float:
        fx0 = f(x0)
        fx1 = f(x1)

        if fx0 * fx1 > 0:
            return -1  # No root in the interval

        # Swap x0 and x1 if necessary to ensure |fx0| >= |fx1|
        if abs(fx0) < abs(fx1):
            x0, x1 = x1, x0
            fx0, fx1 = fx1, fx0

        x2 = x0
        fx2 = fx0
        d = 0
        mflag = True
        steps_taken = 0

        try:
            while steps_taken < max_iter:
                # Ensure that |fx0| >= |fx1|
                if abs(fx0) < abs(fx1):
                    x0, x1 = x1, x0
                    fx0, fx1 = fx1, fx0

                # Compute new point via inverse quadratic interpolation or secant method
                if fx0 != fx2 and fx1 != fx2:
                    L0 = x0 * fx1 * fx2 / ((fx0 - fx1) * (fx0 - fx2))
                    L1 = x1 * fx0 * fx2 / ((fx1 - fx0) * (fx1 - fx2))
                    L2 = x2 * fx1 * fx0 / ((fx2 - fx0) * (fx2 - fx1))
                    new_point = L0 + L1 + L2
                else:
                    new_point = x1 - (fx1 * (x1 - x0) / (fx1 - fx0))

                # Check if the new point is valid or fallback to bisection
                if (new_point < (3 * x0 + x1) / 4 or new_point > x1 or
                    (mflag and abs(new_point - x1) >= abs(x1 - x2) / 2) or
                    (not mflag and abs(new_point - x1) >= abs(x2 - d) / 2) or
                    (mflag and abs(x1 - x2) < tolerance) or
                    (not mflag and abs(x2 - d) < tolerance)):
                    new_point = (x0 + x1) / 2
                    mflag = True
                else:
                    mflag = False

                fnew = f(new_point)
                d = x2
                x2 = x1

                # Update the interval based on the sign of the function at the new point
                if fx0 * fnew < 0:
                    x1 = new_point
                    fx1 = fnew
                else:
                    x0 = new_point
                    fx0 = fnew

                steps_taken += 1

                # Check if we've converged to within the tolerance
                if abs(fnew) < tolerance:
                    return new_point
        except:
            return -1
        
        return -1  # If max iterations reached without convergence

class BloodCapacitance(BaseModelClass):
    '''
    The blood capacitance model is an extension of the Capacitance model.
    The Capacitance model is extended by incorporating routines in the 'volume_in' method which 
    take care of the transport of blood solutes and gasses. It also holds parameters for acidbase and oxygenation routines.
    The solutes in a BloodCapacitance are set by the Blood model during setup and initialization.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties (these parameters are set to the correct value by the ModelEngine)
        self.u_vol: float = 0.0                         # unstressed volume UV of the capacitance in (L)
        self.el_base: float = 0.0                       # baseline elastance E of the capacitance in (mmHg/L)
        self.el_k: float = 0.0                          # non-linear elastance factor K2 of the capacitance (unitless)
        self.pres_ext: float = 0.0                      # external pressure p2(t) (mmHg)
        self.pres_cc: float = 0.0                       # external pressure from chest compressions (mmHg)
        self.pres_mus: float = 0.0                      # external pressure from outside muscles (mmHg)
        self.temp: float = 0.0                          # blood temperature (dgs C)
        self.viscosity: float = 6.0                     # blood viscosity (centiPoise = Pa * s)
        self.solutes: dict = {}                         # dictionary holding all solutes
        self.drugs: dict = {}                           # dictionary holding all drug concentrations
        
        # -> general factors 
        self.ans_activity_factor: float = 1.0           # general ans activity factor

        # -> unstressed volume factors
        self.u_vol_factor: float = 1.0                  # factor changing the unstressed volume
        self.u_vol_scaling_factor: float = 1.0          # factor for scaling the unstressed volume
        self.u_vol_ans_factor: float = 1.0              # factor of the ans model influence on the unstressed volume
        self.u_vol_drug_factor: float = 1.0             # factor of the drug model influence

        # -> elastance factors
        self.el_base_factor: float = 1.0                # factor changing the baseline elastance
        self.el_base_scaling_factor: float = 1.0        # factor for scaling the baseline elastance
        self.el_base_ans_factor: float = 1.0            # factor of the ans model influence on the baseline elastance
        self.el_base_drug_factor: float = 1.0           # factor of the drug model influence

        # -> non-linear elastance factors
        self.el_k_factor: float = 1.0                   # factor changing the non-linear part of the elastance
        self.el_k_scaling_factor: float = 1.0           # factor for scaling the non-linear part of the elastance
        self.el_k_ans_factor: float = 1.0               # factor of the ans model influence on the non-linear part of the elastance
        self.el_k_drug_factor: float = 1.0              # factor of the drug model influence

        # -----------------------------------------------
        # initialize dependent properties
        self.vol: float = 0.0                           # volume v(t) (L)
        self.pres: float = 0.0                          # pressure p1(t) (mmHg)
        self.pres_in: float = 0.0                       # recoil pressure of the elastance (mmHg)
        self.to2: float = 0.0                           # total oxygen concentration (mmol/l)
        self.tco2: float = 0.0                          # total carbon dioxide concentration (mmol/l)
        self.ph: float = -1.0                           # ph (unitless)
        self.pco2: float = -1.0                         # pco2 (mmHg)
        self.po2: float = -1.0                          # po2 (mmHg)
        self.so2: float = -1.0                          # o2 saturation
        self.hco3: float = -1.0                         # bicarbonate concentration (mmol/l)
        self.be: float = -1.0                           # base excess (mmol/l)

        # -----------------------------------------------
        # initialize local properties
        
    def calc_model(self) -> None:
        # Incorporate the scaling factors
        _el_base = self.el_base * self.el_base_scaling_factor
        _el_k_base = self.el_k * self.el_k_scaling_factor
        _u_vol_base = self.u_vol * self.u_vol_scaling_factor

        # Incorporate the other factors which modify the independent parameters
        _el = (
            _el_base
            + (self.el_base_factor - 1) * _el_base
            + (self.el_base_ans_factor - 1) * _el_base * self.ans_activity_factor
            + (self.el_base_drug_factor - 1) * _el_base
        )
        _el_k = (
            _el_k_base
            + (self.el_k_factor - 1) * _el_k_base
            + (self.el_k_ans_factor - 1) * _el_k_base * self.ans_activity_factor
            + (self.el_k_drug_factor - 1) * _el_k_base
        )
        _u_vol = (
            _u_vol_base
            + (self.u_vol_factor - 1) * _u_vol_base
            + (self.u_vol_ans_factor - 1) * _u_vol_base * self.ans_activity_factor
            + (self.u_vol_drug_factor - 1) * _u_vol_base
        )
        
        # calculate the recoil pressure of the capacitance
        self.pres_in = _el_k * math.pow((self.vol - _u_vol),2) + _el * (self.vol - _u_vol)
        
        # calculate the total pressure by incorporating the external pressures
        self.pres = self.pres_in + self.pres_ext + self.pres_cc + self.pres_mus

        # reset the external pressures
        self.pres_ext = 0.0
        self.pres_cc = 0.0
        self.pres_mus = 0.0

    def volume_in(self, dvol: float, comp_from: object) -> None:
        # add volume to the capacitance
        self.vol += dvol

        # return if the volume is zero or lower
        if self.vol <= 0.0:
            return

        # process the gasses o2 and co2
        self.to2 += ((comp_from.to2 - self.to2) * dvol) / self.vol
        self.tco2 += ((comp_from.tco2 - self.tco2) * dvol) / self.vol

        # process the solutes
        for solute, conc in self.solutes.items():
            self.solutes[solute] += ((comp_from.solutes[solute] - conc) * dvol) / self.vol

    def volume_out(self, dvol: float) -> float:
        # remove volume from capacitance
        self.vol -= dvol

        # return if the volume is zero or lower
        if self.vol < 0.0 and self.vol < self.u_vol:
            # store the volume which could not be removed. This is a sign of a problem with the modeling stepsize!!
            _vol_not_removed = -self.vol
            # set he current volume to zero
            self.vol = 0.0
            # return the volume which could not be removed
            return _vol_not_removed
        
        # return zero as all volume in dvol is removed from the capactitance
        return 0.0

class BloodTimeVaryingElastance(BaseModelClass):
    '''
    The blood time-varying elastance model is an extension of the time-varying elastance model.
    The timevarying elastance model is extended by incorporating routines in the 'volume_in' method which 
    take care of the transport of blood solutes and gasses. It also holds parameters for acidbase and oxygenation routines.
    The solutes in a BloodTimeVaryingElastance are set by the Blood model during setup and initialization.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties which will be set when the init_model method is called
        self.u_vol: float = 0.0                         # unstressed volume UV of the capacitance in (L)
        self.el_min: float = 0.0                        # minimal elastance Emin in (mmHg/L)
        self.el_max: float = 0.0                        # maximal elastance emax(n) in (mmHg/L)
        self.el_k: float = 0.0                          # non-linear elastance factor K2 of the capacitance (unitless)
        self.pres_ext: float = 0.0                      # external pressure p2(t) in mmHg
        self.pres_cc: float = 0.0                       # external pressure from chest compressions (mmHg) 
        self.pres_mus: float = 0.0                      # external pressure from outside muscles (mmHg)
        self.temp: float = 0.0                          # blood temperature (dgs C)
        self.viscosity: float = 6.0                     # blood viscosity (centiPoise = Pa * s)
        self.solutes: dict = {}                         # dictionary holding all solutes
        self.drugs: dict = {}                           # dictionary holding all drug concentrations

         # -> general factors 
        self.act_factor: float = 0.0                    # activation factor from the heart model (unitless)
        self.ans_activity_factor: float = 1.0           # general ans activity factor

        # -> unstressed volume factors
        self.u_vol_factor: float = 1.0                  # factor changing the unstressed volume
        self.u_vol_scaling_factor: float = 1.0          # factor for scaling the unstressed volume
        self.u_vol_ans_factor: float = 1.0              # factor of the ans model influence on the unstressed volume
        self.u_vol_drug_factor: float = 1.0             # factor of the drug model influence

        # -> elastance factors
        self.el_min_factor: float = 1.0                 # factor changing the baseline elastance
        self.el_min_scaling_factor: float = 1.0         # factor for scaling the baseline elastance
        self.el_min_ans_factor: float = 1.0             # factor of the ans model influence on the baseline elastance
        self.el_min_drug_factor: float = 1.0            # factor of the drug model influence
        self.el_min_drug_factor: float = 1.0            # factor of the drug model influence
        self.el_min_mob_factor: float = 1.0             # factor of the myocardial oxygen balance model influence

        self.el_max_factor: float = 1.0                 # factor changing the baseline elastance
        self.el_max_scaling_factor: float = 1.0         # factor for scaling the baseline elastance
        self.el_max_ans_factor: float = 1.0             # factor of the ans model influence on the baseline elastance
        self.el_max_drug_factor: float = 1.0            # factor of the drug model influence
        self.el_max_mob_factor: float = 1.0             # factor of the myocardial oxygen balance model influence

        # -> non-linear elastance factors
        self.el_k_factor: float = 1.0                   # factor changing the non-linear part of the elastance
        self.el_k_scaling_factor: float = 1.0           # factor for scaling the non-linear part of the elastance
        self.el_k_ans_factor: float = 1.0               # factor of the ans model influence on the non-linear part of the elastance
        self.el_k_drug_factor: float = 1.0              # factor of the drug model influence

        # -----------------------------------------------
        # initialize dependent properties
        self.vol: float = 0.0                           # volume v(t) (L)
        self.pres: float = 0.0                          # pressure p1(t) (mmHg)
        self.pres_in: float = 0.0                       # pressure compared to atmospheric pressure
        self.to2: float = 0.0                           # total oxygen concentration (mmol/l)
        self.tco2: float = 0.0                          # total carbon dioxide concentration (mmol/l)
        self.ph: float = 0.0                            # ph
        self.pco2: float = 0.0                          # pco2 (mmHg)
        self.po2: float = 0.0                           # po2 (mmHg)
        self.so2: float = 0.0                           # o2 saturation
        self.hco3: float = 0.0                          # bicarbonate concentration (mmol/l)
        self.be: float = 0.0                            # base excess (mmol/l)

        # -----------------------------------------------
        # local variables

    def calc_model(self) -> None:
        # Incorporate the scaling factors
        _el_min_base = self.el_min * self.el_min_scaling_factor
        _el_max_base = self.el_max * self.el_max_scaling_factor
        _el_k_base = self.el_k * self.el_k_scaling_factor
        _u_vol_base = self.u_vol * self.u_vol_scaling_factor

        # Incorporate the other factors which modify the independent parameters
        _el_min = (
            _el_min_base
            + (self.el_min_factor - 1) * _el_min_base
            + (self.el_min_ans_factor - 1) * _el_min_base * self.ans_activity_factor
            + (self.el_min_mob_factor - 1) * _el_min_base
            + (self.el_min_drug_factor - 1) * _el_min_base
        )
        _el_max = (
            _el_max_base
            + (self.el_max_factor - 1) * _el_max_base
            + (self.el_max_ans_factor - 1) * _el_max_base * self.ans_activity_factor
            + (self.el_max_mob_factor - 1) * _el_max_base
            + (self.el_max_drug_factor - 1) * _el_max_base
        )

        _el_k = (
            _el_k_base
            + (self.el_k_factor - 1) * _el_k_base
            + (self.el_k_ans_factor - 1) * _el_k_base * self.ans_activity_factor
            + (self.el_k_drug_factor - 1) * _el_k_base
        )
        _u_vol = (
            _u_vol_base
            + (self.u_vol_factor - 1) * _u_vol_base
            + (self.u_vol_ans_factor - 1) * _u_vol_base * self.ans_activity_factor
            + (self.u_vol_drug_factor - 1) * _u_vol_base
        )

        # calculate the recoil pressure of the time varying elastance using the maximal elastance and minimal elastances
        p_ms = (self.vol - _u_vol) * _el_max
        p_ed = _el_k * math.pow((self.vol - _u_vol),2) + _el_min * (self.vol - _u_vol)
        
        # calculate the current recoil pressure
        self.pres_in = (p_ms - p_ed) * self.act_factor + p_ed
        
        # calculate the total pressure by incorporating the external pressures
        self.pres = self.pres_in + self.pres_ext + self.pres_cc + self.pres_mus
        
        # reset the external pressure
        self.pres_ext = 0.0
        self.pres_cc = 0.0
        self.pres_mus = 0.0

    def volume_in(self, dvol: float, comp_from: object) -> None:
        # add volume to the capacitance
        self.vol += dvol

        # return if the volume is zero or lower
        if self.vol <= 0.0:
            return

        # process the gasses
        self.to2 += ((comp_from.to2 - self.to2) * dvol) / self.vol
        self.tco2 += ((comp_from.tco2 - self.tco2) * dvol) / self.vol

        # process the solutes
        for solute, conc in self.solutes.items():
            self.solutes[solute] += ((comp_from.solutes[solute] - conc) * dvol) / self.vol

    def volume_out(self, dvol: float) -> float:
        # remove volume from capacitance
        self.vol -= dvol

        # return if the volume is zero or lower
        if self.vol < 0.0:
            # store the volume which could not be removed. This is a sign of a problem with the modeling stepsize!!
            _vol_not_removed = -self.vol
            # set he current volume to zero
            self.vol = 0.0
            # return the volume which could not be removed
            return _vol_not_removed
        
        # return zero as all volume in dvol is removed from the capactitance
        return 0.0

class BloodResistor(BaseModelClass):
    '''
    The BloodResistor model is a extension of the Resistor model as described in the paper.
    A BloodResistor model is a connector between two blood containing models (e.g. BloodCapacitance or BloodTimeVaryingElastance) and
    the model determines the flow between the two models it connects.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties
        self.r_for:float  = 1.0                         # forward flow resistance Rf (mmHg/l*s)
        self.r_back: float = 1.0                        # backward flow resistance Rb (mmHg/l*s )
        self.r_k: float = 0.0                           # non linear resistance factor K1 (unitless)
        self.comp_from: str = ""                        # holds the name of the upstream component
        self.comp_to: str = ""                          # holds the name of the downstream component
        self.no_flow: bool = False                      # flags whether flow is allowed across this resistor
        self.no_back_flow: bool = False                  # flags whether backflow is allowed across this resistor
        self.p1_ext: float = 0.0                        # external pressure on the inlet (mmHg)
        self.p2_ext: float = 0.0                        # external pressure on the outlet (mmHg)

        # general factors
        self.ans_activity_factor: float = 1.0           # general ans activity factor

        self.r_factor: float = 1.0                      # factor changing the forward and backward resistance
        self.r_scaling_factor: float = 1.0              # factor for scaling the resistance
        self.r_mob_factor: float = 1.0                  # factor of the myocardial oxygen balance model influence on the resistance
        self.r_ans_factor: float = 1.0                  # factor of the autonomic nervous system model influence on the resistance
        self.r_drug_factor: float = 1.0                 # factor of the drug model influence on the resistance

        self.r_k_factor: float = 1.0                    # factor changing the non-linear part of the resistance
        self.r_k_scaling_factor: float = 1.0            # factor for scaling the non-linear part of the resistance
        self.r_k_ans_factor: float = 1.0                # factor of the autonomic nervous system model on the non-linear part of the resistance
        self.r_k_drug_factor: float = 1.0               # factor of the drug model on the non-linear part of the resistance

        # -----------------------------------------------
        # initialize dependent properties
        self.flow: float = 0.0                          # flow f(t) (L/s)

        # -----------------------------------------------
        # local variables
        self._comp_from: object = {}                    # holds a reference to the upstream component
        self._comp_to: object = {}                      # holds a reference to the downstream component

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # find the up- and downstream components and store the references
        self._comp_from = self._model_engine.models[self.comp_from]
        self._comp_to = self._model_engine.models[self.comp_to]
        
        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self) -> None:
        # get the pressure of the volume containing compartments which this resistor connects and incorporate the external pressures
        _p1_t = self._comp_from.pres + self.p1_ext
        _p2_t = self._comp_to.pres + self.p2_ext

        # reset the external pressures
        self.p1_ext = 0
        self.p2_ext = 0

        # incorporate the scaling factors
        _r_for_base = self.r_for * self.r_scaling_factor
        _r_back_base = self.r_back * self.r_scaling_factor
        _r_k_base = self.r_k * self.r_scaling_factor

        # incorporate all factors influencing this resistor
        _r_for = (
            _r_for_base
            + (self.r_factor - 1) * _r_for_base
            + ((self.r_ans_factor - 1) * _r_for_base) * self.ans_activity_factor
            + (self.r_mob_factor - 1) * _r_for_base
            + (self.r_drug_factor - 1) * _r_for_base
        )

        _r_back = (
            _r_back_base
            + (self.r_factor - 1) * _r_back_base
            + ((self.r_ans_factor - 1) * _r_back_base) * self.ans_activity_factor
            + (self.r_mob_factor - 1) * _r_back_base
            + (self.r_drug_factor - 1) * _r_back_base
        )

        _r_k = (
            _r_k_base
            + (self.r_k_factor - 1) * _r_k_base
            + ((self.r_ans_factor - 1) * _r_k_base) * self.ans_activity_factor
            + (self.r_mob_factor - 1) * _r_k_base
            + (self.r_drug_factor - 1) * _r_k_base
        )

        # make the resistances flow dependent
        _r_for += _r_k * self.flow * self.flow
        _r_back += _r_k * self.flow * self.flow

        # reset the current flow as a new value is coming
        self.flow: float = 0.0

        # return if no flow is allowed across this resistor
        if (self.no_flow):
            return
        
        # calculate the forward flow between two volume containing blood capacitances or blood time varying elastances
        if (_p1_t >= _p2_t):
            self.flow = ((_p1_t - _p2_t) - _r_k * math.pow(self.flow, 2)) / _r_for      # flow L/s
            # update the volumes of the connected components
            vol_not_removed = self._comp_from.volume_out(self.flow * self._t)
            self._comp_to.volume_in((self.flow * self._t) - vol_not_removed, self._comp_from)
            return
        
        # calculate the backward flow between two volume containing blood capacitances or blood time varying elastances
        if (_p1_t < _p2_t and not self.no_back_flow):
            self.flow = ((_p1_t - _p2_t) + _r_k * math.pow(self.flow, 2)) / _r_back
            # update the volumes of the connected components
            vol_not_removed = self._comp_to.volume_out(-self.flow * self._t)
            self._comp_from.volume_in((-self.flow * self._t) - vol_not_removed, self._comp_to)
            return

class BloodValve(BaseModelClass):
    '''
    The BloodValve model is a extension of the Valve model as described in the paper.
    A BloodValve model is a connector between two blood containing models (e.g. BloodCapacitance or BloodTimeVaryingElastance) and
    the model determines the flow between the two models it connects. It generally only allows for forward flow.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties
        self.r_for:float  = 1.0                         # forward flow resistance Rf (mmHg/l*s)
        self.r_back: float = 1.0                        # backward flow resistance Rb (mmHg/l*s )
        self.r_k: float = 0.0                           # non linear resistance factor K1 (unitless)
        self.comp_from: str = ""                        # holds the name of the upstream component
        self.comp_to: str = ""                          # holds the name of the downstream component
        self.no_flow: bool = False                      # flags whether flow is allowed across this valve
        self.no_back_flow: bool = True                  # flags whether backflow is allowed across this valve
        self.p1_ext: float = 0.0                        # external pressure on the inlet (mmHg)
        self.p2_ext: float = 0.0                        # external pressure on the outlet (mmHg)

        # general factors
        self.ans_activity_factor: float = 1.0           # general ans activity factor

        self.r_factor: float = 1.0                      # factor changing the forward and backward valve resistance
        self.r_scaling_factor: float = 1.0              # factor for scaling the valve resistance
        self.r_mob_factor: float = 1.0                  # factor of the myocardial oxygen balance model influence on the valve resistance
        self.r_ans_factor: float = 1.0                  # factor of the autonomic nervous system model influence on the valve resistance
        self.r_drug_factor: float = 1.0                 # factor of the drug model influence on the resistance

        self.r_k_factor: float = 1.0                    # factor changing the non-linear part of the valve resistance
        self.r_k_scaling_factor: float = 1.0            # factor for scaling the non-linear part of the valve resistance
        self.r_k_ans_factor: float = 1.0                # factor of the autonomic nervous system model on the non-linear part of the valve resistance
        self.r_k_drug_factor: float = 1.0               # factor of the drug model on the non-linear part of the valve resistance

        # -----------------------------------------------
        # initialize dependent properties
        self.flow: float = 0.0                          # flow f(t) (L/s)

        # -----------------------------------------------
        # local variables
        self._comp_from: object = {}                    # holds a reference to the upstream component
        self._comp_to: object = {}                      # holds a reference to the downstream component

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # find the up- and downstream components and store the references
        self._comp_from = self._model_engine.models[self.comp_from]
        self._comp_to = self._model_engine.models[self.comp_to]
        
        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self) -> None:
        # get the pressure of the volume containing compartments which this valve connects and incorporate the external pressures
        _p1_t = self._comp_from.pres + self.p1_ext
        _p2_t = self._comp_to.pres + self.p2_ext

        # reset the external pressures
        self.p1_ext = 0
        self.p2_ext = 0

        # incorporate the scaling factors
        _r_for_base = self.r_for * self.r_scaling_factor
        _r_back_base = self.r_back * self.r_scaling_factor
        _r_k_base = self.r_k * self.r_scaling_factor

        # incorporate all factors influencing this valve
        _r_for = (
            _r_for_base
            + (self.r_factor - 1) * _r_for_base
            + ((self.r_ans_factor - 1) * _r_for_base) * self.ans_activity_factor
            + (self.r_mob_factor - 1) * _r_for_base
            + (self.r_drug_factor - 1) * _r_for_base
        )

        _r_back = (
            _r_back_base
            + (self.r_factor - 1) * _r_back_base
            + ((self.r_ans_factor - 1) * _r_back_base) * self.ans_activity_factor
            + (self.r_mob_factor - 1) * _r_back_base
            + (self.r_drug_factor - 1) * _r_back_base
        )

        _r_k = (
            _r_k_base
            + (self.r_k_factor - 1) * _r_k_base
            + ((self.r_ans_factor - 1) * _r_k_base) * self.ans_activity_factor
            + (self.r_mob_factor - 1) * _r_k_base
            + (self.r_drug_factor - 1) * _r_k_base
        )

        # make the resistances flow dependent
        _r_for += _r_k * self.flow * self.flow
        _r_back += _r_k * self.flow * self.flow

        # reset the current flow as a new value is coming
        self.flow: float = 0.0

        # return if no flow is allowed across this valve
        if (self.no_flow):
            return
        
        # calculate the forward flow between two volume containing blood capacitances or blood time varying elastances
        if (_p1_t >= _p2_t):
            self.flow = ((_p1_t - _p2_t) - _r_k * math.pow(self.flow, 2)) / _r_for      # flow L/s
            # update the volumes of the connected components
            vol_not_removed = self._comp_from.volume_out(self.flow * self._t)
            self._comp_to.volume_in((self.flow * self._t) - vol_not_removed, self._comp_from)
            return
        
        # calculate the backward flow between two volume containing blood capacitances or blood time varying elastances
        if (_p1_t < _p2_t and not self.no_back_flow):
            self.flow = ((_p1_t - _p2_t) + _r_k * math.pow(self.flow, 2)) / _r_back
            # update the volumes of the connected components
            vol_not_removed = self._comp_to.volume_out(-self.flow * self._t)
            self._comp_from.volume_in((-self.flow * self._t) - vol_not_removed, self._comp_to)
            return

class BloodDiffusor(BaseModelClass):
    '''
    The BloodDiffusor model handles the diffusion of gasses and solutes between two blood containing models 
    (e.g. BloodCapacitance and BloodTimeVaryingElastance). The diffusion between the gasses
    o2 and co2 are partial pressure driven and the diffusion between solutes is concentration driven.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties which will be set when the init_model method is called
        self.comp_blood1: str = "PLF"                   # name of the first blood containing model
        self.comp_blood2: str = "PLM"                   # name of the second blood containing model
        self.dif_o2: float = 0.01                       # diffusion constant for o2 (mmol/mmHg * s)
        self.dif_co2: float = 0.01                      # diffusion constant for co2 (mmol/mmHg * s)
        self.dif_solutes: dict = {}                     # diffusion constants for the different solutes (mmol/mmol * s)

        # factors
        self.dif_o2_factor: float = 1.0                 # factor influencing the diffusion constant for o2
        self.dif_o2_scaling_factor: float = 1.0         # scaling factor for the diffusion constant for o2
        self.dif_co2_factor: float = 1.0                # factor influencing the diffusion constant for co2
        self.dif_co2_scaling_factor: float = 1.0        # scaling factor for the diffusion constant for co2
        self.dif_solutes_factor: float = 1.0            # factor influencing the diffusion constant for all solutes
        self.dif_solutes_scaling_factor: float = 1.0    # scaling factor for the diffusion constant for all solutes      

        # -----------------------------------------------
        # initialize dependent properties

        # -----------------------------------------------
        # local variables
        self._calc_blood_composition: object = None             # holds a reference to the calc_blood_composition function of the Blood model
        self._comp_blood1: object = None                        # holds a reference to the first blood containing model
        self._comp_blood2: object = None                        # holds a reference to the second blood containing model

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # find the two blood containing models and store a reference
        self._comp_blood1 = self._model_engine.models[self.comp_blood1]
        self._comp_blood2 = self._model_engine.models[self.comp_blood2]

        # store a reference to the calc_blood_composition function of the Blood model
        self._calc_blood_composition = self._model_engine.models["Blood"].calc_blood_composition

        # flag that the model is initialized
        self._is_initialized = True


    def calc_model(self):
        # calculate the blood composition of the blood components in this diffusor as we need the partial pressures for the gasses diffusion
        self._calc_blood_composition(self._comp_blood1)
        self._calc_blood_composition(self._comp_blood2)

        # incorporate the factors
        _dif_o2 = self.dif_o2 * self.dif_o2_scaling_factor * self.dif_o2_factor
        _dif_co2 = self.dif_co2 * self.dif_co2_scaling_factor * self.dif_co2_factor

        # diffuse the gasses where the diffusion is partial pressure driven
        do2 = (self._comp_blood1.po2 - self._comp_blood2.po2) * _dif_o2 * self._t * self.dif_o2_factor
        # update the concentrations
        self._comp_blood1.to2 = ((self._comp_blood1.to2 * self._comp_blood1.vol) - do2) / self._comp_blood1.vol
        self._comp_blood2.to2 = ((self._comp_blood2.to2 * self._comp_blood2.vol) + do2) / self._comp_blood2.vol

        dco2 = (self._comp_blood1.pco2 - self._comp_blood2.pco2) * _dif_co2 * self._t * self.dif_co2_factor
        # update the concentrations
        self._comp_blood1.tco2 = ((self._comp_blood1.tco2 * self._comp_blood1.vol) - dco2) / self._comp_blood1.vol
        self._comp_blood2.tco2 = ((self._comp_blood2.tco2 * self._comp_blood2.vol) + dco2) / self._comp_blood2.vol

        # diffuse the solutes where the diffusion is concentration gradient driven
        for sol, dif in self.dif_solutes.items():
            dif = dif * self.dif_solutes_factor * self.dif_solutes_scaling_factor
            # diffuse the solute which is concentration driven
            dsol = (self._comp_blood1.solutes[sol] - self._comp_blood2.solutes[sol]) * dif * self._t
            # update the concentration
            self._comp_blood1.solutes[sol] = ((self._comp_blood1.solutes[sol] * self._comp_blood1.vol) - dsol) / self._comp_blood1.vol
            self._comp_blood2.solutes[sol] = ((self._comp_blood2.solutes[sol] * self._comp_blood2.vol) + dsol) / self._comp_blood2.vol

class BloodPump(BaseModelClass):
    '''
    The BloodPump model is a BloodCapacitance model which an extension by which it can generate a pressure gradient
    over itself and thereby modeling a blood pump. It does this by exerting pressure on the inlet and outlet BloodResistors 
    connected to the blood pump. For the other parts the BloodPump is exactly the same as a BloodCapacitance model
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties (these parameters are set to the correct value by the ModelEngine)
        self.u_vol: float = 0.0                         # unstressed volume UV of the capacitance in (L)
        self.el_base: float = 0.0                       # baseline elastance E of the capacitance in (mmHg/L)
        self.el_k: float = 0.0                          # non-linear elastance factor K2 of the capacitance (unitless)
        self.pres_ext: float = 0.0                      # external pressure p2(t) (mmHg)
        self.pres_cc: float = 0.0                       # external pressure from chest compressions (mmHg)
        self.pres_mus: float = 0.0                      # external pressure from outside muscles (mmHg)
        self.temp: float = 0.0                          # blood temperature (dgs C)
        self.viscosity: float = 6.0                     # blood viscosity (centiPoise = Pa * s)
        self.solutes: dict = {}                         # dictionary holding all solutes
        self.inlet: str = ""                            # name of the BloodResistor at the inlet of the pump
        self.outlet: str = ""                           # name of the BloodResistor at the outlet of the pump
        self.solutes: dict = {}                         # dictionary holding all solute concentrations
        self.drugs: dict = {}                           # dictionary holding all drug concentrations
        
        # -> general factors 
        self.ans_activity_factor: float = 1.0           # general ans activity factor

        # -> unstressed volume factors
        self.u_vol_factor: float = 1.0                  # factor changing the unstressed volume
        self.u_vol_scaling_factor: float = 1.0          # factor for scaling the unstressed volume
        self.u_vol_ans_factor: float = 1.0              # factor of the ans model influence on the unstressed volume
        self.u_vol_drug_factor: float = 1.0             # factor of the drug model influence

        # -> elastance factors
        self.el_base_factor: float = 1.0                # factor changing the baseline elastance
        self.el_base_scaling_factor: float = 1.0        # factor for scaling the baseline elastance
        self.el_base_ans_factor: float = 1.0            # factor of the ans model influence on the baseline elastance
        self.el_base_drug_factor: float = 1.0           # factor of the drug model influence

        # -> non-linear elastance factors
        self.el_k_factor: float = 1.0                   # factor changing the non-linear part of the elastance
        self.el_k_scaling_factor: float = 1.0           # factor for scaling the non-linear part of the elastance
        self.el_k_ans_factor: float = 1.0               # factor of the ans model influence on the non-linear part of the elastance
        self.el_k_drug_factor: float = 1.0              # factor of the drug model influence

        # -----------------------------------------------
        # initialize dependent properties
        self.vol: float = 0.0                           # volume v(t) (L)
        self.pres: float = 0.0                          # pressure p1(t) (mmHg)
        self.pres_in: float = 0.0                       # recoil pressure of the elastance (mmHg)
        self.to2: float = 0.0                           # total oxygen concentration (mmol/l)
        self.tco2: float = 0.0                          # total carbon dioxide concentration (mmol/l)
        self.ph: float = -1.0                           # ph (unitless)
        self.pco2: float = -1.0                         # pco2 (mmHg)
        self.po2: float = -1.0                          # po2 (mmHg)
        self.so2: float = -1.0                          # o2 saturation
        self.hco3: float = -1.0                         # bicarbonate concentration (mmol/l)
        self.be: float = -1.0                           # base excess (mmol/l)
        self.pump_rpm: float = 0.0                      # pump speed in rotations per minute
        self.pump_mode: int = 0                         # pump mode (0=centrifugal, 1=roller pump)

        # -----------------------------------------------
        # initialize local properties
        self._inlet: object = None                      # holds a reference to the inlet BloodResistor
        self._outlet: object = None                     # holds a reference to the outlet BloodResistor
        
    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # find the inlet and outlet resistors
        self._inlet = self._model_engine.models[self.inlet]
        self._outlet = self._model_engine.models[self.outlet]
        
        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self):
        # Incorporate the scaling factors
        _el_base = self.el_base * self.el_base_scaling_factor
        _el_k_base = self.el_k * self.el_k_scaling_factor
        _u_vol_base = self.u_vol * self.u_vol_scaling_factor

        # Incorporate the other factors which modify the independent parameters
        _el = (
            _el_base
            + (self.el_base_factor - 1) * _el_base
            + (self.el_base_ans_factor - 1) * _el_base * self.ans_activity_factor
            + (self.el_base_drug_factor - 1) * _el_base
        )
        _el_k = (
            _el_k_base
            + (self.el_k_factor - 1) * _el_k_base
            + (self.el_k_ans_factor - 1) * _el_k_base * self.ans_activity_factor
            + (self.el_k_drug_factor - 1) * _el_k_base
        )
        _u_vol = (
            _u_vol_base
            + (self.u_vol_factor - 1) * _u_vol_base
            + (self.u_vol_ans_factor - 1) * _u_vol_base * self.ans_activity_factor
            + (self.u_vol_drug_factor - 1) * _u_vol_base
        )

        # calculate the current recoil pressure of the blood pump
        self.pres_in = _el_k * math.pow((self.vol - _u_vol),2) + _el_base * (self.vol - _u_vol)
        
        # calculate the total pressure
        self.pres = self.pres_in + self.pres_ext + self.pres_cc + self.pres_mus

        # reset the external pressure
        self.pres_ext = 0.0
        self.pres_cc = 0.0
        self.pres_mus = 0.0

        # calculate the pump pressure and apply the pump pressures to the connected resistors
        self.pump_pressure = -self.pump_rpm / 25.0
        if self.pump_mode == 0:
            self._inlet.p1_ext = 0.0
            self._inlet.p2_ext = self.pump_pressure
        else:
            self._outlet.p1_ext = self.pump_pressure
            self._outlet.p2_ext = 0.0

    def volume_in(self, dvol: float, comp_from: object) -> None:
        # add volume to the capacitance
        self.vol += dvol

        # return if the volume is zero or lower
        if self.vol <= 0.0:
            return

        # process the gasses o2 and co2
        self.to2 += ((comp_from.to2 - self.to2) * dvol) / self.vol
        self.tco2 += ((comp_from.tco2 - self.tco2) * dvol) / self.vol

        # process the solutes
        for solute, conc in self.solutes.items():
            self.solutes[solute] += ((comp_from.solutes[solute] - conc) * dvol) / self.vol

    def volume_out(self, dvol: float) -> float:
        # remove volume from capacitance
        self.vol -= dvol

        # return if the volume is zero or lower
        if self.vol < 0.0:
            # store the volume which could not be removed. This is a sign of a problem with the modeling stepsize!!
            _vol_not_removed = -self.vol
            # set he current volume to zero
            self.vol = 0.0
            # signal the user that there's a problem
            print(f"Negative volume error in blood capacitance: {self.name}!")

            # return the volume which could not be removed
            return _vol_not_removed
        
        # return zero as all volume in dvol is removed from the capactitance
        return 0.0

class Gas(BaseModelClass):
    '''
    The Gas model takes care of the gas composition. 
    It sets the gasconcentrations, fractions and partial pressures of gasses on gas containing models (e.g. GasCapacitance) 
    depending on pressure, temperature, humidity and fio2 settings. 
    It also houses a routine used by other models for calculating the gas composition in any gas containing model.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties
        self.pres_atm: float = 760.0                    # atmospheric pressure in mmHg
        self.fio2: float = 0.21                         # fractional O2 concentration
        self.temp: float = 20.0                         # global gas temperature (dgs C)
        self.humidity: float = 0.5                      # global gas humidity (fraction)
        self.humidity_settings: dict = {}               # dictionary holding the initial humidity settings of the gas containing models
        self.temp_settings: dict = {}                   # dictionary holding the initial temperature settings of the gas containing models

        # -----------------------------------------------
        # initialize dependent properties

        # -----------------------------------------------
        # initialize local properties
        self._gas_containing_modeltypes: list = ['GasCapacitance']

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)
        
        # set the atmospheric pressure and global gas temperature in all gas containing model
        for _, model in self._model_engine.models.items():
            if model.model_type in self._gas_containing_modeltypes:
                model.pres_atm = self.pres_atm
                model.temp = self.temp
                model.target_temp = self.temp

        # set the temperatures of the different gas containing components
        for model_name, temp in self.temp_settings.items():
            self._model_engine.models[model_name].temp = temp
            self._model_engine.models[model_name].target_temp = temp

        # set the humidity of the different gas containing components
        for model_name, humidity in self.humidity_settings.items():
            self._model_engine.models[model_name].humidity = humidity

        # calculate the gas composition of the gas containing model types
        for _, model in self._model_engine.models.items():
            if model.model_type in self._gas_containing_modeltypes:
                self.set_gas_composition(model, self.fio2, model.temp, model.humidity)

        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self) -> None:
        pass

    def set_atmospheric_pressure(self, new_pres_atm):
        # store the new atmospheric pressure
        self.pres_atm = new_pres_atm
        # set the atmospheric pressure and global gas temperature in all gas containing model
        for _, model in self._model_engine.models.items():
            if model.model_type in self._gas_containing_modeltypes:
                model.pres_atm = self.pres_atm

    def set_temperature(self, new_temp, sites=['OUT', 'MOUTH']):
        # adjust the temperature in de components stored in the sites parameter
        for site in sites:
            self.temp_settings[site] = float(new_temp)

        # set the temperatures of the different gas containing components
        for model_name, temp in self.temp_settings.items():
            self._model_engine.models[model_name].temp = temp
            self._model_engine.models[model_name].target_temp = temp

    def set_humidity(self, new_humidity, sites=['OUT', 'MOUTH']):
        # adjust the humidity in de components stored in the sites parameter
        for site in sites:
            self.humidity_settings[site] = float(new_humidity)

        # set the humidities of the different gas containing components
        for model_name, humidity in self.humidity_settings.items():
            self._model_engine.models[model_name].humidity = humidity

    def set_fio2(self, new_fio2, sites=['OUT', 'MOUTH']):
        # store the new fio2
        self.fio2 = new_fio2

        # calculate the gas composition of the gas containing model types
        for site in sites:
            # get a reference to the mode component
            m = self._model_engine.models[m]
            # set the new gas composition
            self.set_gas_composition(m, self.fio2, m.temp, m.humidity)
    
    def set_gas_composition(self, gc: object, fio2 = 0.205, temp = 37, humidity = 1.0, fico2 = 0.000392) -> None:
        # define dry air
        _fo2_dry = 0.205
        _fco2_dry = 0.000392
        _fn2_dry = 0.794608
        _fother_dry = 0.0
        _gas_constant = 62.36367

        # calculate the dry gas composition depending on the supplied fio2
        new_fo2_dry = fio2
        new_fco2_dry = fico2
        new_fn2_dry = (_fn2_dry * (1.0 - (fio2 + fico2))) / (1.0 - (_fo2_dry + _fco2_dry))
        new_fother_dry = (_fother_dry * (1.0 - (fio2 + fico2))) / (1.0 - (_fo2_dry + _fco2_dry));

        # make sure the latest pressure is available
        gc.calc_model()

        # get the gas capacitance pressure
        pressure = gc.pres

        # calculate the concentration at this pressure and temperature in mmol/l using the gas law
        gc.ctotal = (pressure / (_gas_constant * (273.15 + temp))) * 1000.0

        # calculate the water vapour pressure, concentration and fraction for this temperature and humidity (0 - 1)
        gc.ph2o = math.pow(math.e, 20.386 - 5132 / (temp + 273)) * humidity
        gc.fh2o = gc.ph2o / pressure
        gc.ch2o = gc.fh2o * gc.ctotal

        # calculate the o2 partial pressure, fraction and concentration
        gc.po2 = new_fo2_dry * (pressure - gc.ph2o)
        gc.fo2 = gc.po2 / pressure
        gc.co2 = gc.fo2 * gc.ctotal

        # calculate the co2 partial pressure, fraction and concentration
        gc.pco2 = new_fco2_dry * (pressure - gc.ph2o)
        gc.fco2 = gc.pco2 / pressure
        gc.cco2 = gc.fco2 * gc.ctotal

        # calculate the n2 partial pressure, fraction and concentration
        gc.pn2 = new_fn2_dry * (pressure - gc.ph2o)
        gc.fn2 = gc.pn2 / pressure
        gc.cn2 = gc.fn2 * gc.ctotal

        # calculate the other gas partial pressure, fraction and concentration
        gc.pother = new_fother_dry * (pressure - gc.ph2o)
        gc.fother = gc.pother / pressure
        gc.cother = gc.fother * gc.ctotal

class GasCapacitance(BaseModelClass):
    '''
    The GasCapacitance model is an extension of the Capacitance model.
    The Capacitance model is extended by methods that take care of the gas transport and composition. Heat and watervapour methods are also
    added tot the GasCapacitance model making it a model which contains a gas volume which can be heated and humidified.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties (these parameters are set to the correct value by the ModelEngine)
        self.u_vol: float = 0.0                         # unstressed volume UV of the capacitance in (L)
        self.el_base: float = 0.0                       # baseline elastance E of the capacitance in (mmHg/L)
        self.el_k: float = 0.0                          # non-linear elastance factor K2 of the capacitance (unitless)
        self.pres_atm: float = 760                      # atmospheric pressure (mmHg)
        self.pres_ext: float = 0.0                      # external pressure p2(t) (mmHg)
        self.pres_cc: float = 0.0                       # external pressure from chest compressions (mmHg)
        self.pres_mus: float = 0.0                      # external pressure from outside muscles (mmHg)
        self.fixed_composition: float = False           # flag whether the gas composition of this capacitance can change

        # -> general factors 
        self.ans_activity_factor: float = 1.0           # general ans activity factor

        # -> unstressed volume factors
        self.u_vol_factor: float = 1.0                  # factor changing the unstressed volume
        self.u_vol_scaling_factor: float = 1.0          # factor for scaling the unstressed volume
        self.u_vol_ans_factor: float = 1.0              # factor of the ans model influence on the unstressed volume
        self.u_vol_drug_factor: float = 1.0             # factor of the drug model influence

        # -> elastance factors
        self.el_base_factor: float = 1.0                # factor changing the baseline elastance
        self.el_base_scaling_factor: float = 1.0        # factor for scaling the baseline elastance
        self.el_base_ans_factor: float = 1.0            # factor of the ans model influence on the baseline elastance
        self.el_base_drug_factor: float = 1.0           # factor of the drug model influence

        # -> non-linear elastance factors
        self.el_k_factor: float = 1.0                   # factor changing the non-linear part of the elastance
        self.el_k_scaling_factor: float = 1.0           # factor for scaling the non-linear part of the elastance
        self.el_k_ans_factor: float = 1.0               # factor of the ans model influence on the non-linear part of the elastance
        self.el_k_drug_factor: float = 1.0              # factor of the drug model influence

        # -----------------------------------------------
        # initialize dependent properties
        self.vol: float = 0.0                           # volume v(t) (L)
        self.pres: float = 0.0                          # pressure p1(t) (mmHg)
        self.pres_in: float = 0.0                       # recoil pressure of the elastance (mmHg)
        self.ctotal: float = 0.0                        # total gas molecule concentration (mmol/l)
        self.co2: float = 0.0                           # oxygen concentration (mmol/l) 
        self.cco2: float = 0.0                          # carbon dioxide concentration (mmol/l)
        self.cn2: float = 0.0                           # nitrogen concentration (mmol/l)
        self.cother: float = 0.0                        # other gasses concentration (mmol/l)
        self.ch2o: float = 0.0                          # watervapour concentration (mmol/l)
        self.target_temp: float = 0.0                   # target temperature (dgs C)
        self.humidity: float = 0.0                      # humditity (fraction)
        self.po2: float = 0.0                           # partial pressure of oxygen (mmHg)
        self.pco2: float = 0.0                          # partial pressure of carbon dioxide (mmHg)
        self.pn2: float = 0.0                           # partial pressure of nitrogen (mmHg)
        self.pother: float = 0.0                        # partial pressure of the other gasses (mmHg)
        self.ph2o: float = 0.0                          # partial pressure of water vapour (mmHg)
        self.fo2: float = 0.0                           # fraction of oxygen of total gas volume
        self.fco2: float = 0.0                          # fraction of carbon dioxide of total gas volume
        self.fn2: float = 0.0                           # fraction of nitrogen of total gas volume
        self.fother: float = 0.0                        # fraction of other gasses of total gas volume
        self.fh2o: float = 0.0                          # fraction of water vapour of total gas volume
        self.temp: float = 0.0                          # blood temperature (dgs C)
        
        # local properties
        self._gas_constant: float = 62.36367            # ideal gas law gas constant (L·mmHg/(mol·K))

    def calc_model(self) -> None:
        # Add heat to the gas
        self.add_heat()

        # Add water vapour to the gas
        self.add_watervapour()

        # Incorporate the scaling factors
        _el_base = self.el_base * self.el_base_scaling_factor
        _el_k_base = self.el_k * self.el_k_scaling_factor
        _u_vol_base = self.u_vol * self.u_vol_scaling_factor

        # Incorporate the other factors which modify the independent parameters
        _el = (
            _el_base
            + (self.el_base_factor - 1) * _el_base
            + (self.el_base_ans_factor - 1) * _el_base * self.ans_activity_factor
            + (self.el_base_drug_factor - 1) * _el_base
        )
        _el_k = (
            _el_k_base
            + (self.el_k_factor - 1) * _el_k_base
            + (self.el_k_ans_factor - 1) * _el_k_base * self.ans_activity_factor
            + (self.el_k_drug_factor - 1) * _el_k_base
        )
        _u_vol = (
            _u_vol_base
            + (self.u_vol_factor - 1) * _u_vol_base
            + (self.u_vol_ans_factor - 1) * _u_vol_base * self.ans_activity_factor
            + (self.u_vol_drug_factor - 1) * _u_vol_base
        )

        # calculate the current recoil pressure of the capacitance
        self.pres_in = _el_k * math.pow((self.vol - _u_vol),2) + _el_base * (self.vol - _u_vol)
        
        # calculate the total pressure
        self.pres = self.pres_in + self.pres_ext + self.pres_cc + self.pres_mus + self.pres_atm

        # reset the external pressure
        self.pres_ext = 0.0
        self.pres_cc = 0.0
        self.pres_mus = 0.0

        # calculate the new gas composition
        self.calc_gas_composition()

    def volume_in(self, dvol: float, comp_from: object) -> None:
        # do not change if this capacitance has a fixed composition
        if self.fixed_composition:
            return
           
        # add volume to the capacitance
        self.vol += dvol

         # change the gas concentrations
        if self.vol > 0.0:
            dco2 = (comp_from.co2 - self.co2) * dvol
            self.co2 = (self.co2 * self.vol + dco2) / self.vol

            dcco2 = (comp_from.cco2 - self.cco2) * dvol
            self.cco2 = (self.cco2 * self.vol + dcco2) / self.vol

            dcn2 = (comp_from.cn2 - self.cn2) * dvol
            self.cn2 = (self.cn2 * self.vol + dcn2) / self.vol

            dch2o = (comp_from.ch2o - self.ch2o) * dvol
            self.ch2o = (self.ch2o * self.vol + dch2o) / self.vol

            dcother = (comp_from.cother - self.cother) * dvol
            self.cother = (self.cother * self.vol + dcother) / self.vol

            # change temperature due to influx of gas
            dtemp = (comp_from.temp - self.temp) * dvol
            self.temp = (self.temp * self.vol + dtemp) / self.vol

    def volume_out(self, dvol: float) -> float:
        # do not change the volume if the composition is fixed
        if self.fixed_composition:
            return 0.0
        
        # remove volume from capacitance
        self.vol -= dvol

        # return if the volume is zero or lower
        if self.vol < 0.0:
            # store the volume which could not be removed. This is a sign of a problem with the modeling stepsize!!
            _vol_not_removed = -self.vol
            # set he current volume to zero
            self.vol = 0.0
            # return the volume which could not be removed
            return _vol_not_removed
        
        # return zero as all volume in dvol is removed from the capactitance
        return 0.0

    def add_heat(self) -> None:
        # calculate a temperature change depending on the target temperature and the current temperature
        dT = (self.target_temp - self.temp) * 0.0005
        self.temp += dT

        # change the volume as the temperature changes
        if self.pres != 0.0 and self.fixed_composition == False:
            # as Ctotal is in mmol/l we have convert it as the gas constant is in mol
            dV = (self.ctotal * self.vol * self._gas_constant * dT) / self.pres
            self.vol += dV / 1000.0

        # guard against negative volumes
        if self.vol < 0:
            self.vol = 0

    def add_watervapour(self):
        # Calculate water vapour pressure at current temperature
        pH2Ot = self.calc_watervapour_pressure()

        # do the diffusion from water vapour depending on the tissue water vapour and gas water vapour pressure
        dH2O = 0.00001 * (pH2Ot - self.ph2o) * self._t
        if self.vol > 0.0:
            self.ch2o = (self.ch2o * self.vol + dH2O) / self.vol

        # as the water vapour also takes volume, that volume this is added to the compliance
        if self.pres != 0.0 and self.fixed_composition == False:
            # as dH2O is in mmol/l we have convert it as the gas constant is in mol
            self.vol += ((self._gas_constant * (273.15 + self.temp)) / self.pres) * (dH2O / 1000.0)

    def calc_watervapour_pressure(self) -> float:
        #   calculate the water vapour pressure in air depending on the temperature
        return math.pow(math.e, 20.386 - 5132 / (self.temp + 273))

    def calc_gas_composition(self):
        # calculate the total gas concentrations
        self.ctotal = self.ch2o + self.co2 + self.cco2 + self.cn2 + self.cother

        # protect against division by zero
        if self.ctotal == 0.0:
            return

        # calculate the partial pressures
        self.ph2o = (self.ch2o / self.ctotal) * self.pres
        self.po2 = (self.co2 / self.ctotal) * self.pres
        self.pco2 = (self.cco2 / self.ctotal) * self.pres
        self.pn2 = (self.cn2 / self.ctotal) * self.pres
        self.pother = (self.cother / self.ctotal) * self.pres

        # calculate the fractions
        self.fh2o = self.ch2o / self.ctotal
        self.fo2 = self.co2 / self.ctotal
        self.fco2 = self.cco2 / self.ctotal
        self.fn2 = self.cn2 / self.ctotal
        self.fother = self.cother / self.ctotal

class GasResistor(BaseModelClass):
    '''
    The GasResistor model is a extension of the Resistor model as described in the paper.
    A GasResistor model is a connector between two gas containing models (e.g. GasCapacitance) and
    the model determines the flow between the two models it connects.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties
        self.r_for:float  = 1.0                         # forward flow resistance Rf (mmHg/l*s)
        self.r_back: float = 1.0                        # backward flow resistance Rb (mmHg/l*s )
        self.r_k: float = 0.0                           # non linear resistance factor K1 (unitless)
        self.comp_from: str = ""                        # holds the name of the upstream component
        self.comp_to: str = ""                          # holds the name of the downstream component
        self.no_flow: bool = False                      # flags whether flow is allowed across this resistor
        self.no_back_flow: bool = False                 # flags whether backflow is allowed across this resistor

        # general factors
        self.ans_activity_factor: float = 1.0           # general ans activity factor

        self.r_factor: float = 1.0                      # factor changing the forward and backward resistance
        self.r_scaling_factor: float = 1.0              # factor for scaling the resistance
        self.r_ans_factor: float = 1.0                  # factor of the autonomic nervous system model influence on the resistance
        self.r_drug_factor: float = 1.0                 # factor of the drug model influence on the resistance

        self.r_k_factor: float = 1.0                    # factor changing the non-linear part of the resistance
        self.r_k_scaling_factor: float = 1.0            # factor for scaling the non-linear part of the resistance
        self.r_k_ans_factor: float = 1.0                # factor of the autonomic nervous system model on the non-linear part of the resistance
        self.r_k_drug_factor: float = 1.0               # factor of the drug model on the non-linear part of the resistance

        # -----------------------------------------------
        # initialize dependent properties
        self.flow: float = 0.0                          # flow f(t) (L/s)

        # -----------------------------------------------
        # local variables
        self._comp_from: object = {}                    # holds a reference to the upstream component
        self._comp_to: object = {}                      # holds a reference to the downstream component

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # find the up- and downstream components and store the references
        self._comp_from = self._model_engine.models[self.comp_from]
        self._comp_to = self._model_engine.models[self.comp_to]
        
        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self) -> None:
        # get the pressure of the volume containing compartments which this resistor connects and incorporate the external pressures
        _p1_t = self._comp_from.pres
        _p2_t = self._comp_to.pres

        # incorporate the scaling factors
        _r_for_base = self.r_for * self.r_scaling_factor
        _r_back_base = self.r_back * self.r_scaling_factor
        _r_k_base = self.r_k * self.r_scaling_factor

        # incorporate all factors influencing this resistor
        _r_for = (
            _r_for_base
            + (self.r_factor - 1) * _r_for_base
            + ((self.r_ans_factor - 1) * _r_for_base) * self.ans_activity_factor
            + (self.r_drug_factor - 1) * _r_for_base
        )

        _r_back = (
            _r_back_base
            + (self.r_factor - 1) * _r_back_base
            + ((self.r_ans_factor - 1) * _r_back_base) * self.ans_activity_factor
            + (self.r_drug_factor - 1) * _r_back_base
        )

        _r_k = (
            _r_k_base
            + (self.r_k_factor - 1) * _r_k_base
            + ((self.r_ans_factor - 1) * _r_k_base) * self.ans_activity_factor
            + (self.r_drug_factor - 1) * _r_k_base
        )

        # make the resistances flow dependent
        _r_for += _r_k * self.flow * self.flow
        _r_back += _r_k * self.flow * self.flow

        # reset the current flow as a new value is coming
        self.flow: float = 0.0

        # return if no flow is allowed across this resistor
        if (self.no_flow):
            return
        
        # calculate the forward flow between two volume containing gas capacitances
        if (_p1_t >= _p2_t):
            self.flow = ((_p1_t - _p2_t) - _r_k * math.pow(self.flow, 2)) / _r_for      # flow L/s
            # update the volumes of the connected components
            vol_not_removed = self._comp_from.volume_out(self.flow * self._t)
            self._comp_to.volume_in((self.flow * self._t) - vol_not_removed, self._comp_from)
            return
        
        # calculate the backward flow between two volume containing gas capacitances
        if (_p1_t < _p2_t and not self.no_back_flow):
            self.flow = ((_p1_t - _p2_t) + _r_k * math.pow(self.flow, 2)) / _r_back
            # update the volumes of the connected components
            vol_not_removed = self._comp_to.volume_out(-self.flow * self._t)
            self._comp_from.volume_in((-self.flow * self._t) - vol_not_removed, self._comp_to)
            return

class GasExchanger(BaseModelClass):
    '''
    The GasExcvhanger model handles the diffusion of gasses between a gas containing model and a blood containing model
    (e.g. GasCapacitance and BloodCapacitance). The diffusion between the gasses o2 and co2 are partial pressure driven.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties
        self.dif_o2: float = 0.0                        # diffusion constant for oxygen (mmol/mmHg * s)
        self.dif_co2: float = 0.0                       # diffusion constant for carbon dioxide (mmol/mmHg * s)
        self.comp_blood: str = ""                       # name of the blood component                
        self.comp_gas: str = ""                         # name of the gas component
        
        # -----------------------------------------------
        # factors
        self.dif_o2_factor: float = 1.0                 # factor modifying the oxygen diffusion constant
        self.dif_o2_scaling_factor: float = 1.0         # factor scaling the oxygen diffusion constant

        self.dif_co2_factor: float = 1.0                # factor modifying the carbon diffusion constant
        self.dif_co2_scaling_factor: float = 1.0        # factor scaling the carbion dioxied diffusion constant

        # -----------------------------------------------
        # initialize dependent properties
        self.flux_o2 = 0.0                              # oxygen flux (mmol)
        self.flux_co2 = 0.0                             # carbon dioxide flux (mmol)

        # -----------------------------------------------
        # local variables
        self._blood: object = None                      # reference to the blood component
        self._gas: object = None                        # reference to the gas component
        self._calc_blood_composition: object = None     # reference to the calc_blood_composition function of the Blood model

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # find the up- and downstream components
        self._blood = self._model_engine.models[self.comp_blood]
        self._gas = self._model_engine.models[self.comp_gas]

        # find a reference to the blood component
        self._calc_blood_composition = self._model_engine.models["Blood"].calc_blood_composition
        
        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self) -> None:
        # set the blood composition of the blood component
        self._calc_blood_composition(self._blood)

         # get the partial pressures and gas concentrations from the components
        po2_blood = self._blood.po2
        pco2_blood = self._blood.pco2
        to2_blood = self._blood.to2
        tco2_blood = self._blood.tco2

        co2_gas = self._gas.co2
        cco2_gas = self._gas.cco2
        po2_gas = self._gas.po2
        pco2_gas = self._gas.pco2

        if self._blood.vol == 0.0:
            return
        
        # calculate the O2 flux from the blood to the gas compartment
        self.flux_o2 = ((po2_blood - po2_gas) * self.dif_o2 * self.dif_o2_factor * self.dif_o2_scaling_factor * self._t)

        # calculate the new O2 concentrations of the gas and blood compartments
        new_to2_blood = (to2_blood * self._blood.vol - self.flux_o2) / self._blood.vol
        if new_to2_blood < 0:
            new_to2_blood = 0.0

        new_co2_gas = (co2_gas * self._gas.vol + self.flux_o2) / self._gas.vol
        if new_co2_gas < 0:
            new_co2_gas = 0.0

        # calculate the CO2 flux from the blood to the gas compartment
        self.flux_co2 = ((pco2_blood - pco2_gas) * self.dif_co2 * self.dif_co2_factor * self.dif_co2_scaling_factor * self._t)

        # calculate the new CO2 concentrations of the gas and blood compartments
        new_tco2_blood = (tco2_blood * self._blood.vol - self.flux_co2) / self._blood.vol
        if new_tco2_blood < 0:
            new_tco2_blood = 0.0

        new_cco2_gas = (cco2_gas * self._gas.vol + self.flux_co2) / self._gas.vol
        if new_cco2_gas < 0:
            new_cco2_gas = 0.0

        # transfer the new concentrations
        self._blood.to2 = new_to2_blood
        self._blood.tco2 = new_tco2_blood
        self._gas.co2 = new_co2_gas
        self._gas.cco2 = new_cco2_gas

class Container(BaseModelClass):
    '''
    The Container model is a Capacitance model which can contain other volume containing models (e.g. BloodCapacitance of GasCapacitance).
    The volume of the Container model is determined by the volumes of the models it contains and the Container will transfer it's pressure back to 
    the models it contains. This allows for modeling of the thoracic cage and pericardium. It also is able to vary it's elastance making it 
    possible to simulate breathing. This makes the Container a very flexible model being a mix between a Capacitance and a TimeVaryingElastance model.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties
        self.u_vol: float = 0.0                         # unstressed volume UV of the capacitance in (L)
        self.el_base: float = 0.0                       # baseline elastance E of the capacitance in (mmHg/L)
        self.el_k: float = 0.0                          # non-linear elastance factor K2 of the capacitance (unitless)
        self.pres_ext: float = 0.0                      # external pressure p2(t) (mmHg)
        self.pres_cc: float = 0.0                       # external pressure from chest compressions (mmHg)
        self.pres_mus: float = 0.0                      # external pressure from outside muscles (mmHg)
        self.vol_extra: float = 0.0                     # additional volume of the container (L)
        self.contained_components: list = 0.0           # list of names of models this Container contains
        self.act_factor: float = 0.0                    # activation factor which can modify the elastance of the container
        
        # -> unstressed volume factors
        self.u_vol_factor: float = 1.0                  # factor changing the unstressed volume
        self.u_vol_scaling_factor: float = 1.0          # factor for scaling the unstressed volume

        # -> elastance factors
        self.el_base_factor: float = 1.0                # factor changing the baseline elastance
        self.el_base_scaling_factor: float = 1.0        # factor for scaling the baseline elastance

        # -> non-linear elastance factors
        self.el_k_factor: float = 1.0                   # factor changing the non-linear part of the elastance
        self.el_k_scaling_factor: float = 1.0           # factor for scaling the non-linear part of the elastance

        # -----------------------------------------------
        # initialize dependent properties
        self.vol: float = 0.0                           # volume v(t) (L)
        self.pres: float = 0.0                          # pressure p1(t) (mmHg)
        self.pres_in: float = 0.0                       # recoil pressure of the elastance (mmHg)

        # -----------------------------------------------
        # initialize local properties
        self._contained_components: list = None         # list of references to the models this container contains

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # store the references to the contained models
        self._contained_components = []
        for c in self.contained_components:
            self._contained_components.append(self._model_engine.models[c])
        
        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self):
        # reset the starting volume to the additional volume of the container 
        self.vol = self.vol_extra

        # get the cummulative volume from all contained models and add it to the volume of the container
        for c in self._contained_components:
            self.vol += c.vol

        # incorporate the scaling factor
        _el_base = self.el_base * self.el_base_scaling_factor
        _el_k_base = self.el_k * self.el_k_scaling_factor
        _u_vol_base = self.u_vol * self.u_vol_scaling_factor

        # incorporate the other factors
        _el = (
            _el_base
            + self.act_factor
            + (self.el_base_factor - 1) * _el_base
        )
        _el_k = (
            _el_k_base
            + (self.el_k_factor - 1) * _el_k_base
        )
        _u_vol = (
            _u_vol_base
            + (self.u_vol_factor - 1) * _u_vol_base
        )

        # calculate the current pressure of the container
        self.pres_in = _el_k * math.pow((self.vol - _u_vol),2) + _el * (self.vol - _u_vol)

        # calculate the total pressure
        self.pres = self.pres_in + self.pres_ext + self.pres_cc + self.pres_mus

        # transfer the container pressure to the contained components
        for c in self._contained_components:
            c.pres_ext += self.pres

        # reset the external pressure
        self.pres_ext = 0.0
        self.act_factor = 0.0

class Heart(BaseModelClass):
    '''
    The Heart model takes care of the activation of the TimeVaryingElastances which model the heart (LA, RA, LV and RV).
    It models the heartrate and ECG timings and builds the activation curve for activating the BloodTimeVaryingElastances.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties
        self.heart_rate_ref: float = 110.0              # reference heart rate (beats/minute)
        self.pq_time: float = 0.1                       # pq time (s)
        self.qrs_time: float = 0.075                    # qrs time (s)
        self.qt_time: float = 0.25                      # qt time (s)
        self.av_delay: float = 0.0005                   # delay in the AV-node (s)
                                  
        self.hr_ans_factor: float = 1.0                 # heart rate factor of the autonomic nervous system
        self.hr_mob_factor: float = 1.0                 # heart rate factor of the myocardial oxygen balance model
        self.hr_temp_factor: float = 1.0                # heart rate factor of the temperature (not implemented yet)
        self.hr_drug_factor: float = 1.0                # heart rate factor of the drug model (not implemneted yet)
        self.ans_activity_factor: float = 1.0           # factor determining the global activity of the autonomic nervous system model
        
        # -----------------------------------------------
        # initialize dependent properties
        self.heart_rate: float = 120.0                  # calculated heart rate (beats/minute)
        self.ncc_ventricular: int = 0                   # ventricular contraction counter   
        self.ncc_atrial: int = 0                        # atrial contraction counter
        self.cardiac_cycle_running: int = 0             # signal whether or not the cardiac cycle is running (0 = not, 1 = cardiac cycle running)
        self.cardiac_cycle_time: float = 0.353          # cardiac cycle time (s)
        self.ecg_signal:float = 0.0                     # ECG signal (mV)

        # -----------------------------------------------
        # local properties
        self._kn: float = 0.579                         # constant of the activation curve
        self._prev_cardiac_cycle_running:int = 0        # previous state of the cardiac cycle running flag   
        self._temp_cardiac_cycle_time = 0.0             # counter of the cardiac cycle time (s)
        self._sa_node_interval: float = 1.0             # sinus node interval (s)
        self._sa_node_timer: float = 0.0                # counter for the sinus node (s)
        self._av_delay_timer:float = 0.0                # counter for av-node (s)
        self._pq_timer: float = 0.0                     # counter for the pq-time (s)
        self._pq_running: bool = False                  # flag whether the pq time is running or not
        self._av_delay_running: bool = False            # flag whether the av-delay is running or not
        self._qrs_timer: float = 0.0                    # counter for the qrs time (s)
        self._qrs_running: bool = False                 # flag whether the qrs time is running or not
        self._ventricle_is_refractory: bool = False     # flag whether the ventricle is refractory
        self._qt_timer: float = 0.0                     # counter for the qt time (s)
        self._qt_running: bool = False                  # flag whether the qt time is running or not
        self._la: object = None                         # reference to the left atrium model
        self._lv: object = None                         # reference to the left ventricle model
        self._ra: object = None                         # reference to the right atrium model
        self._rv: object = None                         # reference to the right ventricle model
        self._coronaries: object = None                 # reference to the coronaries model

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # get a reference to the heart component models
        self._la = self._model_engine.models["LA"]
        self._lv = self._model_engine.models["LV"]
        self._ra = self._model_engine.models["RA"]
        self._rv = self._model_engine.models["RV"]
        self._coronaries = self._model_engine.models["COR"]

        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self):
        # set the previous cardiac cycle flag so we can detect when it's changing
        self._prev_cardiac_cycle_running = self.cardiac_cycle_running

        # calculate the heartrate from the reference value and all other influences
        self.heart_rate = (
            self.heart_rate_ref
            + (self.hr_ans_factor - 1.0) * self.heart_rate_ref * self.ans_activity_factor
            + (self.hr_mob_factor - 1.0) * self.heart_rate_ref
            + (self.hr_temp_factor - 1.0) * self.heart_rate_ref
            + (self.hr_drug_factor - 1.0) * self.heart_rate_ref
        )
        # calculate the qtc time depending on the heartrate
        self.cqt_time = self.calc_qtc(self.heart_rate)

         # calculate the sinus node interval in seconds depending on the heart rate
        self._sa_node_interval = 60.0 / self.heart_rate

        # has the sinus node period elapsed?
        if self._sa_node_timer > self._sa_node_interval:
            # reset the sinus node timer
            self._sa_node_timer = 0.0
            # signal that the pq-time starts running
            self._pq_running = True
            # reset the atrial activation curve counter
            self.ncc_atrial = -1
            # flag that the cardiac cycle is running
            self.cardiac_cycle_running = 1
            # reset the cardiac cycle time
            self._temp_cardiac_cycle_time = 0.0
        
        # has the pq time period elapsed?
        if self._pq_timer > self.pq_time:
            # reset the pq timer
            self._pq_timer = 0.0
            # signal that pq timer has stopped
            self._pq_running = False
            # signal that the av delay timer has started
            self._av_delay_running = True
        
        # has the av delay time elasped
        if self._av_delay_timer > self.av_delay:
            # reset the av delay timer
            self._av_delay_timer = 0.0
            # signal that the av delay has stopped
            self._av_delay_running = False
            # check whether the ventricles are in a refractory state
            if not self._ventricle_is_refractory:
                # signal that the qrs time starts running
                self._qrs_running = True
                # reset the ventricular activation curve
                self.ncc_ventricular = -1

        # has the qrs time period elapsed?
        if self._qrs_timer > self.qrs_time:
            # reset the qrs timer
            self._qrs_timer = 0.0
            # signal that the qrs timer has stopped
            self._qrs_running = False
            # signal that the at timer starts running
            self._qt_running = True
            # signal that the ventricles are now in a refractory state
            self._ventricle_is_refractory = True

        # has the qt time period elapsed?
        if self._qt_timer > self.cqt_time:
            # reset the qt timer
            self._qt_timer = 0.0
            # signal that the qt timer has stopped
            self._qt_running = False
            # signal that the ventricles are coming out of their refractory state
            self._ventricle_is_refractory = False
            # flag that the cardiac cycle has ended
            self.cardiac_cycle_running = 0
            # store the cardiac cycle time
            self.cardiac_cycle_time = self._temp_cardiac_cycle_time

        # increase the timers with the modeling stepsize as set by the model base class
        self._sa_node_timer += self._t

        # check the cardiac cycle
        if self.cardiac_cycle_running == 1:
            self._temp_cardiac_cycle_time += self._t

        # increase the timers depending on the state of the cardiac cycle
        if self._pq_running:
            self._pq_timer += self._t

        if self._av_delay_running:
            self._av_delay_timer += self._t

        if self._qrs_running:
            self._qrs_timer += self._t

        if self._qt_running:
            self._qt_timer += self._t

        # increase the heart activation function counters
        self.ncc_atrial += 1
        self.ncc_ventricular += 1

        # calculate the varying elastance factor
        self.calc_varying_elastance()
 
    def calc_varying_elastance(self):
        # calculate the atrial activation factor
        _atrial_duration = self.pq_time / self._t
        if self.ncc_atrial >= 0 and self.ncc_atrial < _atrial_duration:
            self.aaf = math.sin(math.pi * (self.ncc_atrial / _atrial_duration))
        else:
            self.aaf = 0.0

        # calculate the ventricular activation factor
        _ventricular_duration = (self.qrs_time + self.cqt_time) / self._t
        if self.ncc_ventricular >= 0 and self.ncc_ventricular < _ventricular_duration:
            self.vaf = (
                self.ncc_ventricular / (self._kn * _ventricular_duration)
            ) * math.sin(math.pi * (self.ncc_ventricular / _ventricular_duration))
        else:
            self.vaf = 0.0

        # transfer the activation factor to the heart components
        self._la.act_factor = self.aaf
        self._ra.act_factor = self.aaf
        self._lv.act_factor = self.vaf
        self._rv.act_factor = self.vaf
        self._coronaries.act_factor = self.vaf

    def calc_qtc(self, hr):
        if hr > 10.0:
            # Bazett's formula
            return self.qt_time * math.sqrt(60.0 / hr)
        else:
            return self.qt_time * 2.449

class Breathing(BaseModelClass):
    '''
    The Breathing model takes care of the spontaneous breathing by calculating the respiratory rate and target tidal volume
    from the target minute volume which is provided by the autonomic nervous system. It modifies the elastance of the 
    thoracic cage by setting the act factor of the Thorax model (which is a Container model) parameter in time using the Mecklenburgh function.

    Reference:
    Mecklenburgh JS, al-Obaidi TA, Mapleson WW. 
    A model lung with direct representation of respiratory muscle activity. 
    Br J Anaesth. 1992 Jun;68(6):603-12. doi: 10.1093/bja/68.6.603. PMID: 1610636.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties
        self.breathing_enabled: bool = True                 # flags whether spontaneous breathing is enabled or not
        self.minute_volume_ref: float = 0.2                 # reference minute volume (L/kg/min)
        self.minute_volume_ref_factor: float = 1.0          # factor influencing the reference minute volume
        self.minute_volume_ref_scaling_factor: float = 1.0  # scaling factor of the reference minute volume
        self.vt_rr_ratio: float = 0.0001212                 # ratio between the tidal volume and respiratory rate
        self.vt_rr_ratio_factor: float = 1.0                # factor influencing the ratio between the tidal volume and respiratory rate
        self.vt_rr_ratio_scaling_factor: float = 1.0        # scaling factor of the ratio between the tidal volume and respiratory rate 
        self.rmp_gain_max: float = 50.0                     # maximum elastance change (mmHg/L)
        self.ie_ratio: float = 0.3                          # ratio of the inspiratory and expiratory time
        self.mv_ans_factor: float = 1.0                     # factor influencing the minute volume as set by the autonomic nervous system
        self.ans_activity_factor: float = 1.0               # global factor of the autonomic nervous system activity

        # -----------------------------------------------
        # initialize dependent properties
        self.target_minute_volume: float = 0.0              # target minute volume as set by the autonomic nervous system (L/min/kg)
        self.resp_rate: float = 36.0                        # calculated respiratory rate (breaths/min)
        self.target_tidal_volume: float = 0,0               # calculated target tidal volume (L)
        self.minute_volume: float = 0.0                     # minute volume (L/min)
        self.exp_tidal_volume: float = 0.0                  # expiratory tidal volume (L)
        self.insp_tidal_volume: float = 0.0                 # inspiratory tidal volume (L)
        self.resp_muscle_pressure: float = 0.0              # calculated elastance change (mmHg/L)
        self.ncc_insp: int = 0                              # counter of the inspiratory phase (unitless)
        self.ncc_exp: int = 0                               # counter of the expiratory phase (unitless)
        self.rmp_gain: float = 0.0                          # elastance change gain (mmHg/L)

        # -----------------------------------------------
        # local properties  
        self._eMin4: float = math.pow(math.e, -4)           # constant of the Mecklenburgh function
        self._ti: float = 0.4                               # inspiration time (s)
        self._te: float = 1.0                               # expiration time (s)
        self._breath_timer: float = 0.0                     # time of the current breath (s)
        self._breath_interval: float = 60.0                 # breathing interval (s)
        self._insp_running: float = False                   # flag whether the inspiration is running or not
        self._insp_timer: float = 0.0                       # inspiration timer (s)
        self._temp_insp_volume: float = 0.0                 # volume counter for the inspiratory volume (L)
        self._exp_running: float = False                    # flag whether the expiratio is running or not
        self._exp_timer: float = 0.0                        # expiration timer (s)
        self._temp_exp_volume: float = 0.0                  # volume counter for the expiratory volume (L)

    def calc_model(self):
        # get the current model weight
        _weight = self._model_engine.weight

        # calculate the target minute volume
        _minute_volume_ref = (
            self.minute_volume_ref
            * self.minute_volume_ref_factor
            * self.minute_volume_ref_scaling_factor
            * _weight
        )

        # calculate the target minute volume
        self.target_minute_volume = (_minute_volume_ref + (self.mv_ans_factor - 1.0) * _minute_volume_ref) * self.ans_activity_factor

        # calculate the respiratory rate and target tidal volume from the target minute volume
        self.vt_rr_controller(_weight)

        # calculate the inspiratory and expiratory time
        self._breath_interval = 60.0
        if self.resp_rate > 0:
            self._breath_interval = 60.0 / self.resp_rate
            self._ti = self.ie_ratio * self._breath_interval
            self._te = self._breath_interval - self._ti

        # is it time to start a breath?
        if self._breath_timer > self._breath_interval:
            # reset the breath timer
            self._breath_timer = 0.0
            # flag that the inspiration is running
            self._insp_running = True
            # reset the inspiration timer and counter
            self._insp_timer = 0.0
            self.ncc_insp = 0.0

        # has the inspiration time elapsed?
        if self._insp_timer > self._ti:
            # reset the inspiration timer
            self._insp_timer = 0.0
            # flag that the expiration is running and the inspiration is not
            self._insp_running = False
            self._exp_running = True
            self.ncc_exp = 0.0
            # reset the expiratory volume counter
            self._temp_exp_volume = 0.0
            # store the expiratory volume
            self.insp_tidal_volume = self._temp_insp_volume

        # has the expiration time elapsed?
        if self._exp_timer > self._te:
            # reset the expiration timer
            self._exp_timer = 0.0
            # flag that the expiration is not running anymofre
            self._exp_running = False
            # reset the inspiratory volume counter
            self._temp_insp_volume = 0.0
            # store the expiratory volume
            self.exp_tidal_volume = -self._temp_exp_volume

            # calculate the rmp gain as we might not have reached the target tidal volume
            if self.breathing_enabled:
                # if the target volume is not reached then increase the respiratory muscle gain otherwise lower it
                if abs(self.exp_tidal_volume) < self.target_tidal_volume:
                    self.rmp_gain += 0.1
                if abs(self.exp_tidal_volume) > self.target_tidal_volume:
                    self.rmp_gain -= 0.1
                # guard against zero
                if self.rmp_gain < 0.0:
                    self.rmp_gain = 0.0
                # guard the maximum respiratory muscle gain
                if self.rmp_gain > self.rmp_gain_max:
                    self.rmp_gain = self.rmp_gain_max

            # store the current minute volume
            self.minute_volume = self.exp_tidal_volume * self.resp_rate

        # increase the timers
        self._breath_timer += self._t

        # inpiration
        if self._insp_running:
            # increase the inspiration timer and counter
            self._insp_timer += self._t
            self.ncc_insp += 1
            # increase the inspiratory volume
            if self._model_engine.models["MOUTH_DS"].flow > 0:
                self._temp_insp_volume += (self._model_engine.models["MOUTH_DS"].flow * self._t)

        # expiration
        if self._exp_running:
            # increase the expiration timer and counter
            self._exp_timer += self._t
            self.ncc_exp += 1
            # increase the expiratory volume
            if self._model_engine.models["MOUTH_DS"].flow < 0:
                self._temp_exp_volume += (self._model_engine.models["MOUTH_DS"].flow * self._t)

        # reset the respiratory muscle pressure as this is calculated again in every model run
        self.resp_muscle_pressure = 0.0

        # calculate the new respiratory muscle pressure
        if self.breathing_enabled:
            self.resp_muscle_pressure = self.calc_resp_muscle_pressure()
        else:
            # reset all state variables when not breathing
            self.resp_rate = 0.0
            self.ncc_insp = 0.0
            self.ncc_exp = 0.0
            self.target_tidal_volume = 0.0
            self.resp_muscle_pressure = 0.0

        # transfer the respiratory muscle pressure to the thorax
        self._model_engine.models["THORAX"].act_factor = (self.resp_muscle_pressure * 100.0)

    def vt_rr_controller(self, _weight):
        # calculate the spontaneous resp rate depending on the target minute volume (from ANS) and the set vt-rr ratio
        self.resp_rate = math.sqrt(self.target_minute_volume / (self.vt_rr_ratio * self.vt_rr_ratio_factor * self.vt_rr_ratio_scaling_factor * _weight))

        # calculate the target tidal volume depending on the target resp rate and target minute volume (from ANS)
        if self.resp_rate > 0:
            self.target_tidal_volume = self.target_minute_volume / self.resp_rate
    
    def calc_resp_muscle_pressure(self):
        mp = 0.0
        # inspiration
        if self._insp_running:
            mp = (self.ncc_insp / (self._ti / self._t)) * self.rmp_gain

        # expiration
        if self._exp_running:
            mp = ((math.pow(math.e, -4.0 * (self.ncc_exp / (self._te / self._t))) - self._eMin4) / (1.0 - self._eMin4)) * self.rmp_gain

        return mp
    
    def switch_breathing(self, state):
        # switch on/off the spontaneous breathing
        self.breathing_enabled = state

class Ans(BaseModelClass):
    '''
    The autonomic nervous system model uses Afferent (sensor) and Efferent (effect) pathways to control the respiration and circulation. 
    It does this by taking the outputs of the Afferent (sensor) pathways (normalized receptor firing rate 0 - 1), assign an effect weight to the output 
    and transfer the effect to the associated Efferent (effect) Pathway. 
    Using this very flexible principle the ANS model controls the heart rate, heart contraction, venous volume, systemic and pulmonary vascular resistance 
    and elastance and the minute volume.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # independent properties
        self.ans_active: bool = True                    # flag whether the ans is effect or not
        self.pathways: list = []                        # list of pathways the ANS model uses

        # -----------------------------------------------
        # local properties
        self._update_interval: float = 0.015            # update interval of the ANS, for performance reasons this is slower then the modeling step size (s)
        self._update_counter: float = 0.0               # update counter (s)                   
        self._pathways = {}                             # pathways with the stored references to the Afferent (sensor) and Efferent (effector) pathways
        self._calc_blood_composition = None             # reference to the calc_blood_composition function of the Blood model
        self._ascending_aorta = None                    # reference to the ascending aorta.

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # Initialize the pathways with references to the necessary models
        for pathway in self.pathways:
            self._pathways[pathway["name"]] = {
                # store a reference to the Afferent (sensor)
                "sensor": self._model_engine.models[pathway["sensor"]],
                # store a reference to the Efferent (effector)
                "effector": self._model_engine.models[pathway["effector"]],
                # store whether or not the pathway is activae
                "active": pathway["active"],
                # store the pathway effect weight
                "effect_weight": pathway["effect_weight"],
                # initialize a state variable for the pathway activity
                "pathway_activity": 0.0,
            }

        # Reference the blood composition calculation method
        self._calc_blood_composition = self._model_engine.models["Blood"].calc_blood_composition

        # Reference the models on which the ANS depends
        self._ascending_aorta = self._model_engine.models["AA"]

        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self):
        # return if ans is not active
        if not self.ans_active:
            return
        
        # increase the update counter
        self._update_counter += self._t
        # is it time to run the calculations?
        if self._update_counter >= self._update_interval:
            # reset the update counter
            self._update_counter = 0.0

            # calculate the necessary bloodgasses for the ANS
            self._calc_blood_composition(self._ascending_aorta)
            
            # connect the afferent (sensor) with the efferent (effector)
            for _, pathway in self._pathways.items():
                if pathway["active"]:
                    # get the firing rate from the Afferent pathway
                    _firing_rate_afferent = pathway["sensor"].firing_rate
                    # get the effect size 
                    _effect_size = pathway["effect_weight"]
                    # transfer the receptor firing rate and effect size to the effector
                    pathway["effector"].update_effector(_firing_rate_afferent, _effect_size)

class Afferent(BaseModelClass):
    '''
    The Afferent class models an autonomic nervous system afferent (sensor) pathway. It is a sensor which generates a normalized receptor firing rate (0-1) 
    depending on the value of it's input. It has a setpoint, minimal and maximal value. 
    At the setpoint the firing rate is 0.5, at the maximal value the firing rate 1.0 and 0.0 at the minimal firing rate. 
    It also incorporates a time constant on the changes of the firing rate.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties which will be set when the init_model method is called
        self.input: str = ""                            # name of the input using dot notation (e.g. AA.po2)    
        self.min_value: float = 0.0                     # minimum of the input (firing rate is 0.0)
        self.set_value: float = 0.0                     # setpoint of the input (firing rate is 0.5)
        self.max_value: float = 0.0                     # maximum of the input (firing rate is 1.0)
        self.time_constant: float = 1.0                 # time constant of the firing rate change (s)

        # -----------------------------------------------
        # initialize dependent properties
        self.input_value: float = 0.0                   # input value
        self.firing_rate: float = 0.0                   # normalized receptor firing rate (0 - 1)

        # -----------------------------------------------
        # local properties
        self._update_interval: float = 0.015             # update interval of the receptor (s)
        self._update_counter: float = 0.0                # counter of the update interval (s)
        self._max_firing_rate: float = 1.0               # maximum normalized firing rate 1.0
        self._set_firing_rate: float = 0.5               # setpoint normalized firing rate 0.5
        self._min_firing_rate: float = 0.0               # minimum normalized firing rate 0.0
        self._input_site: object = None                  # reference to the input model
        self._input_prop: str = ""                       # reference to the input property
        self._gain: float = 0.0                          # gain of the firing rate

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # Get a reference to the input site
        model, prop = self.input.split(".")
        self._input_site = self._model_engine.models[model]
        self._input_prop = prop

        # Set the initial values
        self.current_value = getattr(self._input_site, self._input_prop)
        self.firing_rate = self._set_firing_rate
        
        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self) -> None:
        # for performance reasons, the update is done only every 15 ms instead of every step
        self._update_counter += self._t
        if self._update_counter >= self._update_interval:
            self._update_counter = 0.0

            # get the input value
            self.input_value = getattr(self._input_site, self._input_prop)

            # Calculate the activation value
            if self.input_value > self.max_value:
                _activation = self.max_value - self.set_value
            elif self.input_value < self.min_value:
                _activation = self.min_value - self.set_value
            else:
                _activation = self.input_value - self.set_value

            # Calculate the gain
            if _activation > 0:
                # Calculate the gain for positive activation
                self._gain = (self._max_firing_rate - self._set_firing_rate) / (self.max_value - self.set_value)
            else:
                # Calculate the gain for negative activation
                self._gain = (self._set_firing_rate - self._min_firing_rate) / (self.set_value - self.min_value)

            # Calculate the firing rate of the receptor
            _new_firing_rate = self._set_firing_rate + self._gain * _activation

            # Incorporate the time constant to calculate the firing rate
            self.firing_rate = self._update_interval * ((1.0 / self.time_constant) * (-self.firing_rate + _new_firing_rate)) + self.firing_rate

class Efferent(BaseModelClass):
    '''
    The Efferent class models an autonomic nervous system efferent (effect) pathway and can be difficult to understand. 
    It can take in (multiple) normalized firing rates (0-1), calculates the average firing rate and translates this to an effect size on the target by using 
    an activation function with the gain depending on the minimal at maximal effect sizes. The effect size change is also modified by a time constant. 
    The effect size is then transferred to the Effector target by setting the ans factor of the target. 
    
    For example: the effect on the heartrate has an effect_at_max_firing_rate of 0.428 and a effect_at_min_firing_rate of 1.5. 
    This means that the effect size on the heartrate (hr_ans_factor) at an average firing rate of 1.0 is 0.428 and 1.5 at an average firing rate of 0.0.
    This means that when the heart_rate_ref = 110, the heart_rate is about 47 at an avg effector firing rate of 1.0 and 165 at an avg effector firing rate of 0.0
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent parameters
        self.target: str = ""                           # name of the target using dot notation (e.g. Heart.hr_ans_factor) 
        self.effect_at_max_firing_rate: float = 0.0     # effect size at average input firing rate of 1.0
        self.effect_at_min_firing_rate: float = 0.0     # effect size at average input firing rate of 0.0
        self.tc: float = 0.0                            # time constant of the effect change (s)

        # -----------------------------------------------
        # initialize dependent parameters
        self.firing_rate: float = 0.0                   # firing rate (unitless)
        self.effector: float = 1.0                      # current effector size 

        # -----------------------------------------------
        # initialize local parameters
        self._target_model: object = None               # reference to the target model
        self._target_prop: str = ""                     # name of the parameter of the target model
        self._update_interval = 0.015                   # update interval of the effector (s)
        self._update_counter = 0.0                      # update counter (s)
        self._cum_firing_rate: float = 0.0              # cummulative firing rate of the model step
        self._cum_firing_rate_counter = 1.0             # counter for number of inputs

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # get a reference to the target model property
        model, prop = self.target.split(".")
        self._target_model = self._model_engine.models[model]
        self._target_prop = prop

        # flag that the model is initialized
        self._is_initialized = True


    def calc_model(self) -> None:
        # for performance reasons, the update is done only every 15 ms instead of every step
        self._update_counter += self._t
        if self._update_counter >= self._update_interval:
            self._update_counter = 0.0

            # Determine the total average firing rate as the firing rate can be set by multiple pathways
            self.firing_rate = 0.5
            if self._cum_firing_rate_counter > 0.0:
                self.firing_rate = self._cum_firing_rate / self._cum_firing_rate_counter

            # Translate the average firing rate to the effect factor. 
            # If the firing rate is aboven 0.5 use the mxe_high parameter, otherwise use the mxe_low parameter
            if self.firing_rate >= 0.5:
                effector = 1.0 + ((self.effect_at_max_firing_rate - 1.0) / 0.5) * (self.firing_rate - 0.5)
            else:
                effector = self.effect_at_min_firing_rate + ((1.0 - self.effect_at_min_firing_rate) / 0.5) * self.firing_rate

            # Incorporate the time constant for the effector change
            self.effector = (self._update_interval * ((1.0 / self.tc) * (-self.effector + effector)) + self.effector)

            # Transfer the effect factor to the target model
            setattr(self._target_model, self._target_prop, self.effector)

            # Reset the effect factor and number of effectors
            self._cum_firing_rate = 0.5
            self._cum_firing_rate_counter = 0.0

    # update effector firing rate
    def update_effector(self, new_firing_rate, weight) -> None:
        # increase the firing rate with a value determing on the
        self._cum_firing_rate += (new_firing_rate - 0.5) * weight
        self._cum_firing_rate_counter += 1.0

class Metabolism(BaseModelClass):
    '''
    The Metabolism class models the oxygen use and carbon dioxide production in a range of blood containing models. 
    For each of the metabolic active models the Metabolism model has the fraction of the total oxygen use and calculates the change of the to2 and tco2
    in each of the metabolic active models dependeing on that fractional oxygen use and the respiratory quotient determines the co2 production.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties
        self.met_active: bool = True                    # flags whether the metabolism is active or not
        self.vo2: float = 8.1                           # oxygen use in ml/kg/min
        self.vo2_factor: float = 1.0                    # fraction which modulates the oxygen use by outside models
        self.vo2_scaling_factor: float = 1.0            # scaling factor of the oxygen use
        self.resp_q: float = 0.8                        # respiratory quotient for carbon dioxide production
        self.resp_q_scaling_factor: float = 1.0         # scaling factor for respiratory quotient for carbon dioxide production
        self.metabolic_active_models: dict = {}         # dictionary containing key values pairs with the key being the model component and the value the fractional oxygen use

    def calc_model(self) -> None:
        # translate the VO2 in ml/kg/min to VO2 in mmol for this stepsize (assumption is a temperature 37 degrees and atmospheric pressure)
        vo2_step = ((0.039 * self.vo2 * self.vo2_factor * self.vo2_scaling_factor * self._model_engine.weight) / 60.0) * self._t

        for model, fvo2 in self.metabolic_active_models.items():
            # get the vol, tco2 and to2 from the blood compartment
            vol = self._model_engine.models[model].vol
            to2 = self._model_engine.models[model].to2
            tco2 = self._model_engine.models[model].tco2

            if vol == 0.0:
                return
            
            # calculate the change in oxygen concentration in this step
            dto2 = vo2_step * fvo2

            # calculate the new oxygen concentration in blood
            new_to2 = (to2 * vol - dto2) / vol

            # guard against negative values
            if new_to2 < 0:
                new_to2 = 0

            # calculate the change in co2 concentration in this step
            dtco2 = vo2_step * fvo2 * self.resp_q * self.resp_q_scaling_factor

            # calculate the new co2 concentration in blood
            new_tco2 = (tco2 * vol + dtco2) / vol

            # guard against negative values
            if new_tco2 < 0:
                new_tco2 = 0

            # store the new to2 and tco2
            self._model_engine.models[model].to2 = new_to2
            self._model_engine.models[model].tco2 = new_tco2

class Mob(BaseModelClass):
    '''
    The myocardial oxygen balance (Mob) class models the dynamic oxygen use and carbon dioxide production (metabolism) of the heart 
    and models the effect on the heart (heartrate and contractility). 
    
    The metabolism of the heart consists of 4 parts. The basal metabolism (bm), the excitation-contraction coupling (ecc), the potential energy (pe) 
    and the stroke work dependent metabolism (pva). The model is able to lower the basal metabolism in hypoxic circumstances using an activation function.
    
    It calculates the basal metabolism depending on the available po2, the stroke work of a cardiac cycle, calculates the cost of the excitation-contraction coupling and 
    determines the potential energy (pe).
    
    The end product is a myocardial oxygen balance which determines the effect on the heart (rate and contractility)
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # independent properties
        self.mob_active: bool = True                    # flag which determines whether or not the myocardial oxygen balance model is active
        self.to2_min: float = 0.0002                    # minimal oxygen concentration for the basal metabolism (mmol/l)
        self.to2_ref: float = 0.2                       # reference oxygen concentration for the basal metabolism (mmol/l)
        self.resp_q: float = 0.1                        # respiratory quotient for oxygen usse depending carbon dioxide production
        self.bm_vo2_ref: float = 0.0007                 # reference basal oxygen use 
        self.bm_vo2_min: float = 0.00035                # minimal basal oxygen use
        self.bm_vo2_tc: float = 5                       # time constant for the change in basal metabolism in a hypoxic condition (s)
        self.bm_g: float = 0.0
        self.ecc_ref: float = 0.00000301                # reference oxygen use for eletrical coupling
        self.pva_ref: float = 0.00143245                # reference oxygen use for stroke work
        self.pe_ref: float = 0                          # reference oxygen use for potential energy
        self.hr_factor: float = 1                       # current influence on the heart rate (factor)
        self.hr_factor_max: float = 1                   # accelerating influence on the heart rate (so the model does not accelerate the heartrate)
        self.hr_factor_min: float = 0.01                # influen
        self.hr_tc: float = 5                           # time constant of the effect on the heartrate (s)
        self.cont_factor: float = 1                     # current influence on the contractility of the heart (factor)
        self.cont_factor_max: float = 1        
        self.cont_factor_min: float = 0.01
        self.cont_tc: float = 5                         # time constant of the effect on the contractillity of the heart (s)
        self.ans_factor: float = 1                      # current activity of the autonomic nervous system (1=normal activity)
        self.ans_factor_max: float = 1                  # maximal activity of the autonomic nervous system
        self.ans_factor_min: float = 0.01               # minimal activity of the autonomic nervous system
        self.ans_tc: float = 5                          # time constant of the change in activity of the autonomic nervous system
        self.ans_activity_factor: float = 1             # global autonomic nervous system activity (1= normal)

        # dependent properties
        self.hw: float = 0.0      
        self.mob_vo2: float = 0.0
        self.bm_vo2: float = 0.0
        self.ecc_vo2: float = 0.0
        self.pe_vo2: float = 0.0
        self.pva_vo2: float = 0.0
        self.pva: float = 0.0
        self.stroke_work_lv: float = 0.0
        self.stroke_work_rv: float = 0.0

        # local properties and intermediates
        self._cor: object = None
        self._aa_cor: object = None
        self._heart: object = None
        self._lv: object = None
        self._rv: object = None
        self._a_to2: float = 0.0
        self._d_bm_vo2: float = 0.0
        self._d_hr: float = 0.0
        self._d_cont: float = 0.0
        self._d_ans: float = 0.0
        self._ml_to_mmol: float = 22.414
        self._cc_time: float = 0.0
        self._prev_lv_vol: float = 0.0
        self._prev_lv_pres: float = 0.0
        self._prev_rv_vol: float = 0.0
        self._prev_rv_pres: float = 0.0
        self._pv_area_lv: float = 0.0
        self._pv_area_rv: float = 0.0
        self._pv_area_lv_inc: float = 0.0
        self._pv_area_rv_inc: float = 0.0
        self._pv_area_lv_dec: float = 0.0
        self._pv_area_rv_dec: float = 0.0

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # store the references to the models the Mob model needs
        self._aa = self._model_engine.models["AA"]
        self._aa_cor = self._model_engine.models["AA_COR"]
        self._cor = self._model_engine.models["COR"]
        self._heart = self._model_engine.models["Heart"]
        self._lv = self._model_engine.models["LV"]
        self._rv = self._model_engine.models["RV"]
        
        # set the heart weight -> at 3.545 that is about 23 grams
        self.hw = 7.799 + 0.004296 * self._model_engine.weight * 1000.0

        # this gain determines at which rate the baseline vo2 is reduced when the to2 drops below the setpoint
        self.bm_g = (self.bm_vo2_ref * self.hw - self.bm_vo2_min * self.hw) / (self.to2_ref - self.to2_min)
        
        # calculate the gain of the effectors heart rate, contractility and autonomic nervous system suppression
        self.hr_g = (self.hr_factor_max - self.hr_factor_min) / (self.to2_ref - self.to2_min)
        self.cont_g = (self.cont_factor_max - self.cont_factor_min) / (self.to2_ref - self.to2_min)
        self.ans_g = (self.ans_factor_max - self.ans_factor_min) / (self.to2_ref - self.to2_min)

        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self) -> None:
        # do not run of mob is not active
        if not self.mob_active:
            return
        
        # get the necessary model properties from the coronaries
        to2_cor = self._cor.to2
        tco2_cor = self._cor.tco2
        vol_cor = self._cor.vol
        self._cc_time = self._heart.cardiac_cycle_time

        # calculate the activation function of the baseline to2, which is zero when the to2 is above to2 setpoint
        self._a_to2 = self.activation_function(to2_cor, self.to2_ref, self.to2_ref, self.to2_min)

        '''
        If the to2 fals below the to2_ref the baseline metabolism is lowered, 
        the heartrate slows down, the contractility is reduced and the autonomic nervous system influence is inhibited
        as this effect is not instantaneous we need to incorporate the time constants
        '''
        self._d_bm_vo2 = (self._t * ((1 / self.bm_vo2_tc) * (-self._d_bm_vo2 + self._a_to2)) + self._d_bm_vo2)
        self._d_hr = self._t * ((1 / self.hr_tc) * (-self._d_hr + self._a_to2)) + self._d_hr
        self._d_cont = (self._t * ((1 / self.cont_tc) * (-self._d_cont + self._a_to2)) + self._d_cont)
        self._d_ans = self._t * ((1 / self.ans_tc) * (-self._d_ans + self._a_to2)) + self._d_ans

        # calculate the basal metabolism in mmol O2 / sec is dependent on the to2 in the coronary blood
        self.bm_vo2 = self.calc_bm()

        # calculate the effects on the heartrate, contractility and autonomic nervous system inhibition
        self.calc_hypoxia_effects()

        # calculate the energy cost of the excitation-contraction coupling in mmol O2 / cardiac cycle
        self.ecc_vo2 = self.calc_ecc()

        # calculate the pressure volume loop area which is the total stroke work and convert it to mmol O2 / cardiac cycle
        self.pva_vo2 = self.calc_pva()

        # calculate the potentential mechanical work stored in the ventricular wall and convert it to mmol O2 / cardiac cycle
        self.pe_vo2 = self.calc_pe()

        # calculate the total myocardial vo2
        self.mob_vo2 = self.bm_vo2 + self.ecc_vo2 + self.pva_vo2 + self.pe_vo2

        # so the basal metabolism is always running but the pe, ecc and pva are only calculated relevant during a cardiac cycle
        bm_vo2_step = self.bm_vo2 * self._t

        # the ecc_vo2, pva_vo2 are only running during a cardiac cycle which is stored in the heart object cardiac_cycle_time variable
        ecc_vo2_step = 0.0
        pva_vo2_step = 0.0
        pe_vo2_step = 0.0

        if self._cc_time > 0.0 and self._heart.cardiac_cycle_running:
            ecc_vo2_step = (self.ecc_vo2 / self._cc_time) * self._t
            pva_vo2_step = (self.pva_vo2 / self._cc_time) * self._t
            pe_vo2_step = (self.pe_vo2 / self._cc_time) * self._t

        # calculate the total vo2 in mmol O2 for the model step
        self.mvo2_step = bm_vo2_step + ecc_vo2_step + pva_vo2_step + pe_vo2_step

        # calculate the co2 production in this model step
        co2_production = self.mvo2_step * self.resp_q

        # calculate the myocardial oxygen balance in mmol/s
        o2_inflow = self._aa_cor.flow * self._aa.to2
        # in mmol/s
        o2_use = self.mvo2_step / self._t
        # in mmol/s
        self.mob = o2_inflow - o2_use + to2_cor

        # calculate the new blood composition of the coronary blood
        if vol_cor > 0:
            new_to2_cor = (to2_cor * vol_cor - self.mvo2_step) / vol_cor
            new_tco2_cor = (tco2_cor * vol_cor + co2_production) / vol_cor
            if new_to2_cor >= 0:
                self._cor.to2 = new_to2_cor
                self._cor.tco2= new_tco2_cor

    def calc_bm(self) -> float:
        # calculate the baseline vo2 in ml O2 / cardiac cycle (about 20% of total vo2 of myocardium in steady state)
        bm_vo2 = (self.bm_vo2_ref * self.hw + self._d_bm_vo2 * self.bm_g)
       
        # chech that the basal metabolism doesn't fall below the defined lower limit
        if bm_vo2 < (self.bm_vo2_min * self.hw):
            bm_vo2 = (self.bm_vo2_min * self.hw) 

        # return the basal metabolism vo2 in mmol O2 / cardiac cycle
        return bm_vo2 / self._ml_to_mmol

    def calc_ecc(self) -> float:
        # calculate the excitation contraction coupling in mmol O2 / cardiac cycle relates to the costs of ion transport and calcium cycling
        self.ecc_lv = self._lv.el_max
        self.ecc_rv = self._rv.el_max
        self.ecc = (self.ecc_lv + self.ecc_rv) / 1000.0

        return (self.ecc * self.ecc_ref  * self.hw) / self._ml_to_mmol
        # is about 15% in steady state;

    def calc_pe(self) -> float:
        # calculate the potential mechanical work stored in the ventricular wall in mmol O2 / cardiac cycle which does not have a direct metabolic cost but is stored energy
        self.pe = 0

        return (self.pe * self.pe_ref * self.hw) / self._ml_to_mmol

    def calc_pva(self) -> float:
        # detect the start of the cardiac cycle and calculate the area of the pv loop of the previous cardiac cycle
        if (self._heart.cardiac_cycle_running and not self._heart._prev_cardiac_cycle_running):
            # calculate the stroke work of the ventricles (l * mmHg/cardiac cycle)
            self.stroke_work_lv = self._pv_area_lv_dec - self._pv_area_lv_inc
            self.stroke_work_rv = self._pv_area_rv_dec - self._pv_area_rv_inc

            # reset the counters
            self._pv_area_lv_inc = 0.0
            self._pv_area_rv_inc = 0.0
            self._pv_area_lv_dec = 0.0
            self._pv_area_rv_dec = 0.0

        # calculate the pv area of this model step
        _dV_lv = self._lv.vol - self._prev_lv_vol
        # if the volume is increasing count the stroke volume
        if _dV_lv > 0:
            self._pv_area_lv_inc += (_dV_lv * self._prev_lv_pres + (_dV_lv * (self._lv.pres - self._prev_lv_pres)) / 2.0)
        else:
            self._pv_area_lv_dec += (-_dV_lv * self._prev_lv_pres + (-_dV_lv * (self._lv.pres - self._prev_lv_pres)) / 2.0)

        _dV_rv = self._rv.vol - self._prev_rv_vol
        # if the volume is increasing count the stroke volume
        if _dV_rv > 0:
            self._pv_area_rv_inc += (_dV_rv * self._prev_rv_pres + (_dV_rv * (self._rv.pres - self._prev_rv_pres)) / 2.0)
        else:
            self._pv_area_rv_dec += (-_dV_rv * self._prev_rv_pres + (-_dV_rv * (self._rv.pres - self._prev_rv_pres)) / 2.0)

        # store current volumes and pressures
        self._prev_lv_vol = self._lv.vol
        self._prev_lv_pres = self._lv.pres

        self._prev_rv_vol = self._rv.vol
        self._prev_rv_pres = self._rv.pres

        # return the total pressure volume area of both ventricles
        self.pva = self.stroke_work_lv + self.stroke_work_rv

        # return and calculate the pva_vo2 part
        return (self.pva * self.pva_ref * self.hw) / self._ml_to_mmol

    def calc_hypoxia_effects(self) -> None:
        # when hypoxia gets severe the ANS influence gets inhibited and the heartrate, contractility and baseline metabolism are decreased

        # calculate the new ans activity (1.0 is max activity and 0.0 is min activity) which controls the ans activity
        self.ans_activity_factor = 1.0 + self.ans_g * self._d_ans
        self._heart.ans_activity_factor = self.ans_activity_factor

        # calculate the mob factor which controls the heart rate
        self.hr_factor = 1.0 + self.hr_g * self._d_hr
        self._heart.hr_mob_factor = self.hr_factor

        # calculate the mob factor which controls the contractility of the heart
        self.cont_factor = 1.0 + self.cont_g * self._d_cont
        self._heart._lv.el_max_mob_factor = self.cont_factor
        self._heart._rv.el_max_mob_factor = self.cont_factor
        self._heart._la.el_max_mob_factor = self.cont_factor
        self._heart._ra.el_max_mob_factor = self.cont_factor

    def activation_function(self, value, max, setpoint, min) -> float:
        activation = 0.0

        if value >= max:
            activation = max - setpoint
        else:
            if value <= min:
                activation = min - setpoint
            else:
                activation = value - setpoint

        return activation

class Ventilator(BaseModelClass):
    '''
    The Ventilator class models a mechanical ventilator and showcases the powerful object oriented design of explain. 
    It uses GasCapacitances to model the inside air compartment of the ventilator, the tubing and the outside air. 
    The in- and expiratory valves are modeled using a GasResistor. Pressure regulated volume control, pressure control, pressure support 
    are mechanical ventilator modes which are supported. Triggering of the mechanical breath is also supported 
    but interfacing with the Breathing model. The model generates pressure, volume and flow signals and 
    inspiratory and expiratory tidal volumes, minute volumes, end tidal co2, co2 curves, resistance
    and compliance measurements.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize the independent properties
        self.pres_atm: float = 760                      # atmospheric pressure (mmHg)
        self.fio2: float = 0.205                        # inspiratory fraction of oxygen
        self.humidity: float = 1.0                      # humidity (percentage / 100%)
        self.temp: float = 37                           # temperature (dgs C)
        self.ettube_diameter: float = 4                 # endotracheal tube diameter (mm)
        self.ettube_length: float = 110                 # endotracheal tube length (mm)
        self.vent_mode: str = "PRVC"                    # ventilator mode (PC/PRVS/PS)                
        self.vent_rate: float = 40                      # ventilator rate (breaths/min)
        self.tidal_volume: float = 0.015                # target tidal volume (L)
        self.insp_time: float = 0.4                     # inspiration time (s)
        self.insp_flow: float = 12                      # inspiratory flow (L/min)
        self.exp_flow: float = 3                        # expiratory flow (L/min)
        self.pip_cmh2o: float = 14                      # peak inspiratory pressure (cmH2O)
        self.pip_cmh2o_max: float = 14                  # maximal peak inspiratory pressure (cmH2O)
        self.peep_cmh2o: float = 3                      # positive end expiratory pressure (cmH2O)
        self.trigger_volume_perc: float = 6             # trigger volume percentage of tidal volume (%)
        self.synchronized: bool = False                 # flag whether ventilator is synchronized with breathing or not       

        # -----------------------------------------------
        # initialize the dependent properties
        self.pres: float = 0.0                          # pressure at beginning of endotracheal tube (cmH2O)
        self.flow: float = 0.0                          # flow from the tubing to the endotracheal tube (L/s)
        self.vol: float = 0.0                           # volume measured at the beginning of the endotracheal tube (L)
        self.exp_time: float = 1.0                      # expiration time (s)
        self.trigger_volume: float = 0.0                # trigger volume (L)
        self.exp_tidal_volume: float = 0.0              # expiratory tidal volume (L)
        self.insp_tidal_volume: float = 0.0             # inspiratory tidal volume (L)
        self.ncc_insp: float = 0.0                      # inspiration counter (unitless)
        self.ncc_exp: float = 0.0                       # expiration counter (unitless)
        self.etco2: float = 0.0                         # end tidal carbon dioxide partial pressure (mmHg)
        self.co2: float = 0.0                           # carbon dioxide partial pressure at endotracheal tube (mmHg)                       
        self.triggered_breath: bool = False             # flag whether the current breath is triggered or not
    
        # -----------------------------------------------
        # initialize the local properties
        self._vent_gasin: object = None                 # reference to the GasCapacitance model of the air compartment of the ventilator at inspiration side
        self._vent_gascircuit: object = None            # reference to the GasCapacitance modeling the ventilator tubing              
        self._vent_gasout: object = None                # reference to the GasCapacitance modeing the air compartment of the ventilator at the expiration side
        self._vent_insp_valve: object = None            # reference to the GasResistor modeling the inspiration valve
        self._vent_exp_valve: object= None              # reference to the GasResistor modeling the expiration valve
        self._vent_ettube: object = None                # reference to the GasResistor modeling the endotracheal tube
        self._set_gas_composition = None                # reference to routine in the GasModel which calculates gas composition
        self._ventilator_parts: list = []               # list holding all ventilator model parts
        self._ettube_length_ref: float = 110            # reference value of the endotracheal tube length used to calculate the resistance of the tube
        self._pip: float = 0.0                          # peak inspiratory pressure (mmHg)
        self._pip_max: float = 0.0                      # maximal peak inspiratory pressure (mmHg)
        self._peep: float = 0.0                         # positive end expiratory pressure (mmHg)
        self._a: float = 0.0                            # parameter used for endotracheal tube resistance calculation
        self._b: float= 0.0                             # parameter used for endotracheal tube resistance calculation
        self._insp_time_counter: float = 0.0            # counter of the inspiration time (s)
        self._exp_time_counter: float = 0.0             # counter of the expiration time (s)
        self._insp_tidal_volume_counter: float = 0.0    # counter of the inspiratory volume (L)
        self._exp_tidal_volume_counter: float = 0.0     # counter of the expiratory volume (L)
        self._trigger_volume_counter: float = 0.0       # counter of the trigger volume (L)
        self._inspiration: bool = False                 # flags whether the inspiration is running or not
        self._expiration: bool = True                   # flags whether the expiration is running or not
        self._tv_tolerance: float = 0.0005              # sets the allowed tolerance between the target and measurement tidal volume (L)
        self._trigger_blocked: bool = False             # flags whether a trigger is blocked or not
        self._trigger_start: bool = False               # flags whether a trigger is started
        self._breathing_model: object = None            # reference to the breathing model for spontaneous breath detection
        self._peak_flow: float = 0.0                    # holds the maximal generated flow (L/s)
        self._prev_et_tube_flow: float = 0.0            # holds the flow across the endotracheal tube of the previous model step (L/s)
        self._et_tube_resistance: float = 40.0          # holds the calculated endotracheal tube resistance (mmHg/l*s)
         
    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # find the breathing model for breath triggering
        self._breathing_model = self._model_engine.models["Breathing"]

        # find the ventilator components
        self._vent_gasin = self._model_engine.models["VENT_GASIN"]
        self._vent_gascircuit = self._model_engine.models["VENT_GASCIRCUIT"]
        self._vent_gasout = self._model_engine.models["VENT_GASOUT"]
        self._vent_insp_valve = self._model_engine.models["VENT_INSP_VALVE"]
        self._vent_ettube = self._model_engine.models["VENT_ETTUBE"]
        self._vent_exp_valve = self._model_engine.models["VENT_EXP_VALVE"]

        # add the ventilator components to a list
        self._ventilator_parts = []
        self._ventilator_parts.append(self._vent_gasin)
        self._ventilator_parts.append(self._vent_gascircuit)
        self._ventilator_parts.append(self._vent_gasout)
        self._ventilator_parts.append(self._vent_insp_valve)
        self._ventilator_parts.append(self._vent_ettube)
        self._ventilator_parts.append(self._vent_exp_valve)

        # get reference to the gas composition calculator
        self._set_gas_composition = self._model_engine.models["Gas"].set_gas_composition

        # set the temperature, atmospheric pressure and humidity of the ventilator parts and calculate the initial composition
        self._vent_gasin.pres_atm = self.pres_atm
        self._vent_gasin.temp = self.temp
        self._vent_gasin.target_temp = self.temp
        self._vent_gasin.humidity = self.humidity
        self._set_gas_composition(self._vent_gasin, self.fio2, self.temp, self.humidity)

        self._vent_gascircuit.pres_atm = self.pres_atm
        self._vent_gascircuit.temp = self.temp
        self._vent_gascircuit.target_temp = self.temp
        self._vent_gascircuit.humidity = self.humidity
        self._set_gas_composition(self._vent_gascircuit, self.fio2, self.temp, self.humidity)

        self._vent_gasout.pres_atm = self.pres_atm
        self._vent_gasout.temp = 20.0
        self._vent_gasout.target_temp = 20.0
        self._vent_gasout.humidity = 0.5
        self._set_gas_composition(self._vent_gasout, 0.205, 20.0, 0.5)

        # set the endotracheal tube diameter
        self.set_ettube_diameter(self.ettube_diameter)

        # set the flow dependent endotracheal tube resistance
        self._et_tube_resistance = self.calc_ettube_resistance(self.flow)
        
        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self) -> None:
        # as the ventilator setting are in cmH2O we need to convert them to mmHg
        self._pip = self.pip_cmh2o / 1.35951
        self._pip_max = self.pip_cmh2o_max / 1.35951
        self._peep = self.peep_cmh2o / 1.35951

        # if the ventilator is set to synchronized we have to look for trigegred breaths
        if self.synchronized:
            self.triggering()

         # call the correct ventilation mode
        if self.vent_mode == "PC" or self.vent_mode == "PRVC":
            self.time_cycling()
            self.pressure_control()

        if self.vent_mode == "PS":
            self.flow_cycling()
            self.pressure_control()

        # calculated the pressure in the ventilator tubing and convert it to cmH2O for reporting
        self.pres = (self._vent_gascircuit.pres - self.pres_atm) * 1.35951

        # report the current flow between the ventilator tubing and the endotracheal tube in L/min
        self.flow = self._vent_ettube.flow * 60.0

        # report the accumulated volume of the current breath in L
        self.vol += self._vent_ettube.flow * 1000 * self._t

        # report the current co2 of the dead-space in mmHg
        self.co2 = self._model_engine.models["DS"].pco2
        
        # calculate the current minute volume in L/min
        self.minute_volume = self.exp_tidal_volume * self.vent_rate

        # calculate the flow dependent endotracheal tube resistance
        self._et_tube_resistance = self.calc_ettube_resistance(self.flow)

    def triggering(self) -> None:
        # determine the trigger volume
        self.trigger_volume = (self.tidal_volume / 100.0) * self.trigger_volume_perc

        # is there a triggered breath (breaths start with ncc_insp ==1), we get this information from the breathing model
        if self._breathing_model.ncc_insp == 1 and not self._trigger_blocked:
            self._trigger_start = True

        # if a spontaneous breath is started start counting the volume
        if self._trigger_start:
            self._trigger_volume_counter += self._vent_ettube.flow * self._t

        # if the trigger volume is exceeded trigger a breath
        if self._trigger_volume_counter > self.trigger_volume:
            # reset the trigger volume counter
            self._trigger_volume_counter = 0.0
            # start a new ventilator assisted breath
            self._exp_time_counter = self.exp_time
            # prepare for a next triggered breath
            self._trigger_start = False
            # flag a triggered breath
            self.triggered_breath = True

    def flow_cycling(self) -> None:
        # in flow cycling mode the ventilator stops the inspiration phase when the inspiratory 
        # flow is decreased to about 70% of the peak inspiratory flow

        # is there flow moving to the lungs and the breath is triggered
        if self._vent_ettube.flow > 0.0 and self.triggered_breath:
            # check whether the flow is increasing
            if self._vent_ettube.flow > self._prev_et_tube_flow:
                # if increasing then keep inspiration going
                self._inspiration = True
                self._expiration = False
                # reset the ncc counter
                self.ncc_insp = -1
                # determine the peak flow
                if self._vent_ettube.flow > self._peak_flow:
                    self._peak_flow = self._vent_ettube.flow

                # store the current exhaled tidal volume
                self.exp_tidal_volume = -self._exp_tidal_volume_counter
            else:
                # if decreasing wait until it is 70% of the peak flow
                if self._vent_ettube.flow < 0.3 * self._peak_flow:
                    # go into expiration
                    self._inspiration = False
                    self._expiration = True
                    # reset the ncc counter
                    self.ncc_exp = -1
                    # reset the tidal volume counter
                    self._exp_tidal_volume_counter = 0.0
                    # reset the triggered breath flag
                    self.triggered_breath = False

            self._prev_et_tube_flow = self._vent_ettube.flow

        if self._vent_ettube.flow < 0.0 and not self.triggered_breath:
            self._peak_flow = 0.0
            self._prev_et_tube_flow = 0.0
            self._inspiration = False
            self._expiration = True
            # reset the ncc counter
            self.ncc_exp = -1
            # calculate the expiratory tidal volume
            self._exp_tidal_volume_counter += self._vent_ettube.flow * self._t

        # if in inspiration increase the timer
        if self._inspiration:
            self.ncc_insp += 1
            self._trigger_blocked = True

        # if in expiration increase the timer
        if self._expiration:
            self.ncc_exp += 1
            self._trigger_blocked = False

    def time_cycling(self) -> None:
        # in time cycling the inspiration and expiration phase have a specific duration

        # calculate the expiration time
        self.exp_time = 60.0 / self.vent_rate - self.insp_time

        # has the inspiration time elapsed?
        if self._insp_time_counter > self.insp_time:
            # reset the inspiration time counter
            self._insp_time_counter = 0.0
            # store the inspiratory tidal volume
            self.insp_tidal_volume = self._insp_tidal_volume_counter
            # reset the inspiratory tidal volume counter
            self._insp_tidal_volume_counter = 0.0
            # flag the inspiration and expiration
            self._inspiration = False
            self._expiration = True
            # reset the triggered breath flag
            self.triggered_breath = False
            # reset the ncc counter
            self.ncc_exp = -1

        # has the expiration time elapsed or is the expiration phase ended by a triggered breath
        if self._exp_time_counter > self.exp_time:
            # reset the expiration time counter
            self._exp_time_counter = 0.0
            # flag the inspiration and expiration
            self._inspiration = True
            self._expiration = False
            # reset the ncc counter
            self.ncc_insp = -1
            # reset the volume counter
            self.vol = 0.0
            # reset the volume counters
            self.exp_tidal_volume = -self._exp_tidal_volume_counter
            # store the end tidal co2
            self.etco2 = self._model_engine.models["DS"].pco2
            # calculate the tidal volume per kg
            self.tv_kg = (self.exp_tidal_volume * 1000.0) / self._model_engine.weight
            # calculate the compliance of the lung
            if self.exp_tidal_volume > 0:
                self.compliance = 1 / (((self._pip - self._peep) * 1.35951) / (self.exp_tidal_volume * 1000.0)) # in ml/cmH2O
            # reset the expiratory tidal volume counter
            self._exp_tidal_volume_counter = 0.0
            # check whether the ventilator is in PRVC mode because we need to adjust the pressure depending on the tidal volume
            if self.vent_mode == "PRVC":
                self.pressure_regulated_volume_control()

        # if in inspiration increase the timer
        if self._inspiration:
            self._insp_time_counter += self._t
            self.ncc_insp += 1
            self._trigger_blocked = True
            self._trigger_volume_counter = 0.0

        # if in expiration increase the timer
        if self._expiration:
            self._exp_time_counter += self._t
            self.ncc_exp += 1
            self._trigger_blocked = False

    def pressure_control(self) -> None:
        # pressure controle ventilation is a time-cycled pressure limited ventilation mode.

        if self._inspiration:
            # close the expiration valve and open the inspiration valve
            self._vent_exp_valve.no_flow = True
            self._vent_insp_valve.no_flow = False

            # prevent back flow to the ventilator
            self._vent_insp_valve.no_back_flow = True

            # set the resistance of the inspiration valve
            self._vent_insp_valve.r_for = (self._vent_gasin.pres + self._pip - self.pres_atm - self._peep) / (self.insp_flow / 60.0)

            # guard the inspiratory pressure
            if self._vent_gascircuit.pres > self._pip + self.pres_atm:
                # close the inspiratory valve when the pressure is reached
                self._vent_insp_valve.no_flow = True

            # calculate the inspiratory tidal volume
            if self._vent_ettube.flow > 0:
                self._insp_tidal_volume_counter += self._vent_ettube.flow * self._t

        if self._expiration:
            # close the inspiration valve and open the expiration valve and prevent backflow
            self._vent_insp_valve.no_flow = True

            self._vent_exp_valve.no_flow = False
            self._vent_exp_valve.no_back_flow = True

            # set the resistance of the expiration valve to and calculate the pressure in the expiration block
            # to simulatie the positive end expiratory pressure 
            self._vent_exp_valve.r_for = 10
            self._vent_gasout.vol = (self._peep / self._vent_gasout.el_base + self._vent_gasout.u_vol)

            # calculate the expiratory tidal volume
            if self._vent_ettube.flow < 0:
                self._exp_tidal_volume_counter += self._vent_ettube.flow * self._t
    
    def pressure_regulated_volume_control(self) -> None:
         # pressure regulated volume control ventilation is a time-cycled pressure limited ventilation mode.
         # where the ventilator increases or decreases the inspiratory pressure depending on whether or not
         # the target volume is reached or not.

        # if the target expiratory tidal volume is not reached then increase the peak inspiratory pressure
        if self.exp_tidal_volume < self.tidal_volume - self._tv_tolerance:
            self.pip_cmh2o += 1.0
            # if the peak inspiratory pressure exceeds the set maximal pressure then don't increase it
            if self.pip_cmh2o > self.pip_cmh2o_max:
                self.pip_cmh2o = self.pip_cmh2o_max

        # if the target expiratory tidal volume is exceeded then decrease the peak inspiratory pressure
        if self.exp_tidal_volume > self.tidal_volume + self._tv_tolerance:
            self.pip_cmh2o -= 1.0
            # if the peak inspiratory pressure falls below the PEEP + 2 don't decrease it any further
            if self.pip_cmh2o < self.peep_cmh2o + 2.0:
                self.pip_cmh2o = self.peep_cmh2o + 2.0

    def switch_ventilator(self, state: bool) -> None:
        # switch the calculations of the ventilator model
        self.is_enabled = state

        # enable of disable all ventilator components
        for vp in self._ventilator_parts:
            vp.is_enabled = state
            # make sure the no flow flag is set correctly on the gas resistors
            if hasattr(vp, "no_flow"):
                vp.no_flow = not state

        # intubate the patient, meaning close MOUTH_DS connector as the ventilator is now connected to the DS component
        self._model_engine.models["MOUTH_DS"].no_flow = state

    def calc_ettube_resistance(self, flow) -> float:
        # calculate the flow dependent endotracheal tube resistance
        _ettube_length_ref = 110
        res = (self._a * flow + self._b) * (self.ettube_length / _ettube_length_ref)
        if res < 15.0:
            res = 15

        # set the resistance 
        self._vent_ettube.r_for = res
        self._vent_ettube.r_back = res

        return res

    def set_ettube_length(self, new_length) -> None:
        if new_length >= 50:
            self.ettube_length = new_length
           
    def set_ettube_diameter(self, new_diameter) -> None:
        if new_diameter > 1.5:
            self.ettube_diameter = new_diameter
            # set resistance parameters (emprical derived from lab)
            self._a = -2.375 * new_diameter + 11.9375
            self._b = -14.375 * new_diameter + 65.9374

    def set_fio2(self, new_fio2) -> None:
        # set the new fio2 and make sure that is doesn't matter how it is provided
        if new_fio2 > 20:
            # user means %
            self.fio2 = new_fio2 / 100.0
        else:
            self.fio2 = new_fio2

        # calculate the new ventilator gas compostition
        self._set_gas_composition(self._vent_gasin, self.fio2, self._vent_gasin.temp, self._vent_gasin.humidity)
    
    def set_humidity(self, new_humidity) -> None:
        if new_humidity >= 0 and new_humidity <= 1.0:
            self.humidity = new_humidity
            # calculate the new ventilator gas compostition
            self._set_gas_composition(self._vent_gasin, self.fio2, self._vent_gasin.temp, self.humidity)

    def set_temp(self, new_temp) -> None:
        self.temp = new_temp
        # calculate the new ventilator gas compostition
        self._set_gas_composition(self._vent_gasin, self.fio2, self.temp, self._vent_gasin.humidity)

    def set_pc(self, pip=14.0, peep=4.0, rate=40.0, t_in=0.4, insp_flow=10.0) -> None:
        # switch to pressure control mode
        self.pip_cmh2o = pip
        self.pip_cmh2o_max = pip
        self.peep_cmh2o = peep
        self.vent_rate = rate
        self.insp_time = t_in
        self.insp_flow = insp_flow
        self.vent_mode = "PC"

    def set_prvc(self, pip_max=18.0, peep=4.0, rate=40.0, tv=15.0, t_in=0.4, insp_flow=10.0) -> None:
        # switch to pressure regulated volume control mode
        self.pip_cmh2o_max = pip_max
        self.peep_cmh2o = peep
        self.vent_rate = rate
        self.insp_time = t_in
        self.tidal_volume = tv / 1000.0
        self.insp_flow = insp_flow
        self.vent_mode = "PRVC"

    def set_psv(self, pip=14.0, peep=4.0, rate=40.0, t_in=0.4, insp_flow=10.0) -> None:
        # switch to pressure support mode
        self.pip_cmh2o = pip
        self.pip_cmh2o_max = pip
        self.peep_cmh2o = peep
        self.vent_rate = rate
        self.insp_time = t_in
        self.insp_flow = insp_flow
        self.vent_mode = "PS"

    def trigger_breath(self, pip=14.0, peep=4.0, rate=40.0, t_in=0.4, insp_flow=10.0):
        # trigger a breath by setting the expiration time counter
        self._exp_time_counter = self.exp_time + 0.1

class Ecls(BaseModelClass):
    '''
    The ECLS class models an ECLS system where blood is taken out of the circulation and oxygen added and carbon dioxied removed
    and then pumpen back into the circulation. It is build with standard Explain components (BloodCapacitances, BloodResistors, BloodPump, GasCapacitances, GasResistor and GasExchanger)
    It showcases the flexibility of the object oriented design of the explain model.
    The cannulas the drain and return blood intro the circulation are a BloodResisters while the tubing and oxygenator are BloodCapacitances
    The pump is modeled by the BloodPump class while eveything is connected by a set of BloodResistors
    The gas part is modelend by a GasCapacitance modeling the gas source, artificial lung and a GasCapacitance modeling the outside air. 
    The GasCapacitances are connected by GasResistors and a GasExchanger model connects the blood and gas parts of the oxygenator.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize the independent parameters
        self.pres_atm: float = 760.0                    # atmospheric pressure (mmHg)
        self.tubing_diameter: float = 0.25              # tubing diameter (inch)
        self.tubing_elastance: float = 11600            # tubing elastance (mmHg/l)
        self.tubing_in_length: float = 1.0              # tubing in length (m)
        self.tubing_out_length: float = 1.0             # tubing out length (m)
        self.tubing_clamped: bool = True                # states whether the tubing is clamped or not
        self.drainage_cannula_diameter: float = 12      # drainage cannula diameter in (Fr)
        self.drainage_cannula_length: float = 0.11      # drainage cannula length (m)
        self.return_cannula_diameter: float = 10        # drainage cannula diameter in (Fr)
        self.return_cannula_length: float = 0.11        # drainage cannula length (m)
        self.pump_volume: float = 0.8                   # volume of the pumphead (l)
        self.oxy_volume: float = 0.8                    # volume of the oxygenator (l)
        self.oxy_resistance: float = 100                # resistance of the oxygenator (mmHg/l*s)
        self.oxy_dif_o2: float = 0.001                  # oxygenator oxygen diffusion constant (mmol/mmHg)
        self.oxy_dif_co2: float = 0.001                 # oxygenator carbon dioxide diffusion constant (mmol/mmHg)
        self.sweep_gas: float = 0.5                     # gas flowing through the gas part of the oxygenator (L/min)
        self.fio2_gas: float = 0.3                      # fractional oxygen content of the oxygenator gas
        self.co2_gas_flow: float = 0.4                  # added carbon dioxide gas flow (L/min)
        self.temp_gas: float = 37.0                     # temperature of the oxygenator gas (dgs C)
        self.humidity_gas: float = 0.5                  # humidity of the oxygenator gas (0-1)
        self.pump_rpm: float = 1500                     # rotations of the centrifugal pump (rpm)

        # initialize the dependent parameters
        self.blood_flow: float = 0.0                    # ecls blood flow (L/min)   
        self.gas_flow: float = 0.0                      # ecls gas flow (L/min)
        self.p_ven: float = 0.0                         # pressure on the drainage side of the ecls system (mmHg)
        self.p_int: float = 0.0                         # pressure between the pump and oxygenator (mmHg)
        self.p_art: float = 0.0                         # pressure after the oxygenator on the return side of the ecls system (mmHg)
        self.pre_oxy_bloodgas: object = {}              # object holding the bloodgas pre-oxygenator
        self.post_oxy_bloodgas: object = {}             # object holding the bloodgas post-oxygenator

        # initialize the local parameters
        self._ecls_tubin = None                         # reference to the inlet tubing on the drainage side (BloodCapacitance)
        self._ecls_pump = None                          # reference to the the ecls pump (BloodPump)
        self._ecls_oxy = None                           # reference to the oxygenator (BloodCapacitance)
        self._ecls_tubout = None                        # reference to the outlet tubing on the return side (BloodCapacitance)
        self._ecls_drainage = None                      # reference to the drainage cnanula (BloodResistor)
        self._ecls_tubin_pump = None                    # reference to the connector between inlet tubing and the pump (BloodResistor)
        self._ecls_pump_oxy = None                      # reference to the connector between the pump and oxygenator (BloodResistor)
        self._ecls_oxy_tubout = None                    # reference to the connector between the oxygenator and outlet tubing (BloodResistor)
        self._ecls_return = None                        # reference to the return cannula (BloodResistor)
        self._ecls_gasin = None                         # reference to the gas source (GasCapacitance)
        self._ecls_gasoxy = None                        # reference to the oxygenator (GasCapacitance)
        self._ecls_gasout = None                        # reference to the gas outside world (GasCapacitance)
        self._ecls_gasin_oxy = None                     # reference to the connector connecting the gas source to the oxygenator (GasResistor)
        self._ecls_oxy_gasout = None                    # reference to the connector connecting the oxygenator to the outside world (Gasresistor)
        self._ecls_parts = []                           # list holding all ecls parts
        self._set_gas_composition = None                # reference to the function of the Gas model which calculates the composition of a gas containing model
        self._calc_blood_composition = None             # reference to the function of the Blood model which calculates the composition of a blood containing model
        self._fico2_gas: float = 0.0004                 # fractional carbon dioxide content of the ecls gas
        self._update_interval = 0.015                   # update interval of the mdoel (s)
        self._update_counter = 0.0                      # update counter (s)
        self._bloodgas_interval = 1.0                   # interval at which the blood gasses are calculated (s)
        self._bloodgas_counter = 0.0                    # counter of the bloodgas interval (s)

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # get a reference to all ecls components for performance reasons
        self._ecls_tubin = self._model_engine.models["ECLS_TUBIN"]
        self._ecls_pump = self._model_engine.models["ECLS_PUMP"]
        self._ecls_oxy = self._model_engine.models["ECLS_OXY"]
        self._ecls_tubout = self._model_engine.models["ECLS_TUBOUT"]
        self._ecls_drainage = self._model_engine.models["ECLS_DRAINAGE"]
        self._ecls_tubin_pump = self._model_engine.models["ECLS_TUBIN_PUMP"]
        self._ecls_pump_oxy = self._model_engine.models["ECLS_PUMP_OXY"]
        self._ecls_oxy_tubout = self._model_engine.models["ECLS_OXY_TUBOUT"]
        self._ecls_return = self._model_engine.models["ECLS_RETURN"]
        self._ecls_gasin = self._model_engine.models["ECLS_GASIN"]
        self._ecls_gasoxy = self._model_engine.models["ECLS_GASOXY"]
        self._ecls_gasout = self._model_engine.models["ECLS_GASOUT"]
        self._ecls_gasin_oxy = self._model_engine.models["ECLS_GASIN_OXY"]
        self._ecls_oxy_gasout = self._model_engine.models["ECLS_OXY_GASOUT"]
        self._ecls_gasex = self._model_engine.models["ECLS_GASEX"]
        
        # clear the ecls part list
        self._ecls_parts = []
        # add the ecls models to the part list
        self._ecls_parts.append(self._ecls_tubin)
        self._ecls_parts.append(self._ecls_pump)
        self._ecls_parts.append(self._ecls_oxy)
        self._ecls_parts.append(self._ecls_tubout)
        self._ecls_parts.append(self._ecls_drainage)
        self._ecls_parts.append(self._ecls_tubin_pump)
        self._ecls_parts.append(self._ecls_pump_oxy)
        self._ecls_parts.append(self._ecls_oxy_tubout)
        self._ecls_parts.append(self._ecls_return)
        self._ecls_parts.append(self._ecls_gasin)
        self._ecls_parts.append(self._ecls_gasoxy)
        self._ecls_parts.append(self._ecls_tubin)
        self._ecls_parts.append(self._ecls_gasout)
        self._ecls_parts.append(self._ecls_gasin_oxy)
        self._ecls_parts.append(self._ecls_oxy_gasout)
        self._ecls_parts.append(self._ecls_gasex)

        # get a reference to the gas and blood composition routines
        self._calc_blood_composition = self._model_engine.models["Blood"].calc_blood_composition
        self._set_gas_composition = self._model_engine.models["Gas"].set_gas_composition

        # set the tubing properties
        self.set_tubing_properties()

        # set the properties of the cannulas
        self.set_cannula_properties()

        # set the properties of the oxygenator
        self.set_oxygenator_properties()

        # set the properties of the pump
        self.set_pump_properties()

        # set the properties of the gas source
        self.set_gas_source_properties()

        # set the properties of the gas outlet
        self.set_gas_outlet_properties()

        # set the properties of the gas oxy
        self.set_gas_oxy_properties()

        # set the gas exchanger properties
        self.set_gasexchanger_properties(self.oxy_dif_o2, self.oxy_dif_co2)

        # set the gas flow
        self.set_gas_flow(self.sweep_gas)

        # set the pump speed
        self.set_pump_speed(1500)

        # set the clamp
        self.set_clamp(True)

        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self) -> None:
        # increase the update timer
        self._update_counter += self._t
        if self._update_counter > self._update_interval:
            self._update_counter = 0.0
            # store the blood and gas flows
            self.blood_flow: float = self._ecls_return.flow * 60.0
            self.gas_flow: float = self._ecls_oxy_gasout.flow * 60.0
            
            # store the pressures
            self.p_ven: float = self._ecls_tubin.pres_in
            self.p_int: float = self._ecls_oxy.pres_in
            self.p_art: float = self._ecls_tubout.pres_in

        # calculate the bloodgasses
        self._bloodgas_counter += self._t
        if self._bloodgas_counter > self._bloodgas_interval:
            self._bloodgas_counter: float = 0.0

            # calculate the bloodgases
            self._calc_blood_composition(self._ecls_tubin)
            self._calc_blood_composition(self._ecls_tubout)

            self.pre_oxy_bloodgas: object = {
                "ph": self._ecls_tubin.ph,
                "po2": self._ecls_tubin.po2,
                "pco2": self._ecls_tubin.pco2,
                "hco3": self._ecls_tubin.hco3,
                "be": self._ecls_tubin.be,
                "so2": self._ecls_tubin.so2
            }
            self.post_oxy_bloodgas: object = {
                "ph": self._ecls_tubout.ph,
                "po2": self._ecls_tubout.po2,
                "pco2": self._ecls_tubout.pco2,
                "hco3": self._ecls_tubout.hco3,
                "be": self._ecls_tubout.be,
                "so2": self._ecls_tubout.so2
            }

    def switch_ecls(self, state: bool) -> None:
        # switch the calculations of the ecls model
        self.is_enabled = state

        # enable of disable all ecls components
        for ep in self._ecls_parts:
            ep.is_enabled = state
            # make sure the no flow flag is set correctly on the resistors (safety)
            if hasattr(ep, "no_flow"):
                ep.no_flow = not state
    
    def set_pump_speed(self, new_pump_speed) -> None:
        # store new pump speed
        self.pump_rpm = new_pump_speed
        # set the pump speed on the BloodPump model
        self._ecls_pump.pump_rpm = self.pump_rpm

    def set_clamp(self, state) -> None:
        # clamp/unclamp the drainage and return tubing by setting the no_flow property
        self.tubing_clamped = state
        self._ecls_drainage.no_flow = state
        self._ecls_return.no_flow = state

    def set_fio2(self, new_fio2) -> None:
        # set the new fio2 of the ecls gas
        if new_fio2 > 20:
            self.fio2_gas = new_fio2 / 100.0
        else:
            self.fio2_gas = new_fio2

        # determine the fico2 of the gas
        self._fico2_gas = (self.co2_gas_flow * 0.001) / self.sweep_gas
        # recalculate the gas composition
        self._set_gas_composition(self._ecls_gasin, self.fio2_gas, self.temp_gas, self.humidity_gas, self._fico2_gas)

    def set_co2_flow(self, new_co2_flow) -> None:
        # store the new gas flow
        self.co2_gas_flow = new_co2_flow
        # determine the fico2 of the gas
        self._fico2_gas = (self.co2_gas_flow * 0.001) / self.sweep_gas
        # recalculate the gas composition
        self._set_gas_composition(self._ecls_gasin, self.fio2_gas, self.temp_gas, self.humidity_gas, self._fico2_gas)
    
    def set_gas_flow(self, new_sweep_gas) -> None:
        # set the new gas flow
        if new_sweep_gas > 0.0:
            # store the new flow
            self.sweep_gas = new_sweep_gas
            # calculate the pressures in the GasCapacitances of the gas part of the ecls system
            self._ecls_gasin.calc_model()
            self._ecls_gasoxy.calc_model()
            self._ecls_gasout.calc_model()
            # calculate the resistances of the connectors which are needed to get the correct gas flow
            self._ecls_gasin_oxy.r_for = (self._ecls_gasin.pres - self.pres_atm) / (self.sweep_gas / 60.0)
            self._ecls_gasin_oxy.r_back = self._ecls_gasin_oxy.r_for

    def set_oxygenator_volume(self, new_volume) -> None:
        # set the new oxygenator volume (blood part)
        self.oxy_volume = new_volume
        # reset the oxygenator
        self.set_oxygenator_properties()

    def set_pump_volume(self, new_volume) -> None:
        # set the new pump volume
        self.pump_volume = new_volume
        # reset the pump
        self.set_pump_volume()

    def set_oxygenator_resistance(self, new_resistance) -> None:
        # set the new oxygnator (blood part) additional resistance (on top of the tubing resistance)
        self.oxy_resistance = new_resistance
        self._ecls_oxy_tubout.r_for = self.calc_tube_resistance(self.tubing_diameter * 0.0254, self.tubing_out_length) + self.oxy_resistance
        self._ecls_oxy_tubout.r_back = self._ecls_oxy_tubout.r_for + self.oxy_resistance
        
    def set_drainage_cannula_diameter(self, new_diameter) -> None:
        # set the drainage cannula diameter
        self.drainage_cannula_diameter = new_diameter
        # reset the dfainage cannula
        self.set_cannula_properties()

    def set_return_cannula_diameter(self, new_diameter) -> None:
        # set the return cannula diameter
        self.return_cannula_diameter = new_diameter
        # reset the return cannula
        self.set_cannula_properties()

    def set_drainage_cannula_length(self, new_length) -> None:
        # set the drainage cannula length
        self.drainage_cannula_length = new_length
        # reset the dfainage cannula
        self.set_cannula_properties()

    def set_return_cannula_length(self, new_length) -> None:
        # set the return cannula length
        self.return_cannula_length = new_length
        # reset the return cannula
        self.set_cannula_properties()

    def set_tubing_diameter(self, new_diameter) -> None:
        # set the tubing diameter
        self.tubing_diameter = new_diameter
        self.set_tubing_properties()

    def set_tubing_length(self, new_length) -> None:
        # set the tubing total length and assume the drainage and return part have the same length
        self.tubing_in_length = new_length / 2.0
        self.tubing_out_length = new_length / 2.0
        self.set_tubing_properties

    def set_tubing_elastance(self, new_elastance) -> None:
        # set the tubing elastance
        self.tubing_elastance = new_elastance
        self.set_tubing_properties()
    
    def set_gas_oxy_properties(self) -> None:
        # calculate the gas composition of the gas part of the oxygenator
        self._fico2_gas = (self.co2_gas_flow * 0.001) / self.sweep_gas
        self._set_gas_composition(self._ecls_gasoxy, self.fio2_gas, self.temp_gas, self.humidity_gas, self._fico2_gas)

    def set_gas_outlet_properties(self) -> None:
        # calculate the composition of the outside air
        self._set_gas_composition(self._ecls_gasout, 0.205, 20.0, 0.1, 0.0004)

    def set_gas_source_properties(self) -> None:
        # calculate the gas composition of the gas source
        self._fico2_gas = (self.co2_gas_flow * 0.001) / self.sweep_gas
        self._set_gas_composition(self._ecls_gasin, self.fio2_gas, self.temp_gas, self.humidity_gas, self._fico2_gas)

    def set_gasexchanger_properties(self, _oxy_dif_o2, _oxy_dif_co2) -> None:
        # set the diffusion constants for oxygen and carbon dioxide of the GasExchanger
        self.oxy_dif_o2 = _oxy_dif_o2
        self.oxy_dif_co2 = _oxy_dif_co2
        self._ecls_gasex.dif_o2 = self.oxy_dif_o2
        self._ecls_gasex.dif_co2 = self.oxy_dif_co2

    def set_oxygenator_properties(self) -> None:
        self._ecls_oxy.vol = self.oxy_volume
        self._ecls_oxy.u_vol = self.oxy_volume
        self._ecls_oxy.calc_model()

    def set_pump_properties(self) -> None:
        self._ecls_pump.vol = self.pump_volume
        self._ecls_pump.u_vol = self.pump_volume
        self._ecls_pump.calc_model()

    def set_cannula_properties(self) -> None:
        _drainage_res = self.calc_tube_resistance(self.drainage_cannula_diameter * 0.00033, self.drainage_cannula_length)
        self._ecls_drainage.r_for = _drainage_res
        self._ecls_drainage.r_back = _drainage_res

        _return_res = self.calc_tube_resistance(self.return_cannula_diameter * 0.00033, self.return_cannula_length)
        self._ecls_return.r_for = _return_res
        self._ecls_return.r_back = _return_res

    def set_tubing_properties(self) -> None:
        # tubing in properties
        _tubing_volume_in = self.calc_tube_volume(self.tubing_diameter * 0.0254, self.tubing_in_length)
        self._ecls_tubin.vol = _tubing_volume_in
        self._ecls_tubin.u_vol = _tubing_volume_in
        self._ecls_tubin.el_base = self.tubing_elastance
        self._ecls_tubin.calc_model()
        self._ecls_tubin_pump.r_for = self.calc_tube_resistance(self.tubing_diameter * 0.0254, self.tubing_in_length)
        self._ecls_tubin_pump.r_back = self._ecls_tubin_pump.r_for

        _tubing_volume_out = self.calc_tube_volume(self.tubing_diameter * 0.0254, self.tubing_out_length)
        self._ecls_tubout.vol = _tubing_volume_out
        self._ecls_tubout.u_vol = _tubing_volume_out
        self._ecls_tubout.el_base = self.tubing_elastance
        self._ecls_tubout.calc_model()
        self._ecls_oxy_tubout.r_for = self.calc_tube_resistance(self.tubing_diameter * 0.0254, self.tubing_out_length) + self.oxy_resistance
        self._ecls_oxy_tubout.r_back = self._ecls_oxy_tubout.r_for + + self.oxy_resistance

    def calc_tube_volume(self, diameter, length) -> float:
        # return the volume in liters
        return math.pi * math.pow(0.5 * diameter, 2) * length * 1000.0
    
    def calc_tube_resistance(self, diameter, length, viscosity=6.0) -> float:
        # resistance is calculated using Poiseuille's Law : R = (8 * n * L) / (PI * r^4)

        # resistance is in mmHg * s / l
        # L = length in meters
        # r = radius in meters
        # n = viscosity in centiPoise

        # convert viscosity from centiPoise to Pa * s
        n_pas = viscosity / 1000.0

        # convert the length to meters
        length_meters = length

        # calculate radius in meters
        radius_meters = diameter / 2

        # calculate the resistance    Pa *  / m3
        res = (8.0 * n_pas * length_meters) / (math.pi * math.pow(radius_meters, 4))

        # convert resistance of Pa/m3 to mmHg/l
        res = res * 0.00000750062
        return res

class Placenta(BaseModelClass):
    '''
    The Placenta class models the placental circulation and gasexchange using core models of the explain model.
    The umbilical arteries and veins are modeled by BloodResistors connected to the descending aorta (DA) and
    inferior vena cava (IVCI). The fetal (PLF) and maternal placenta (PLM) are modeled by twp BloodCapacitances. 
    A BloodDiffusor model instance takes care of the GasExchange between the PLF and PLM.
    '''

    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent parameters
        self.umb_art_diameter: float = 0.005            # diameter of the umbilical arteries combined (m)
        self.umb_art_length: float = 0.5                # length of the umbilical arteries (m)
        self.umb_ven_diameter: float = 0.005            # diameter of the umbilical vein (m)
        self.umb_ven_length: float = 0.5                # length of the umbilical vein (m)
        self.umb_art_res: float = 30000                 # resistance of the umbilical artery (mmHg/L *s)
        self.umb_ven_res: float = 30000                 # resistance of the umbilical vein (mmHg/L * s)
        self.plf_u_vol: float = 0.15                    # unstressed volume of the fetal placenta (L)
        self.plf_el_base: float = 5000.0                # elastance of the fetal placenta (mmHg/L)
        self.plm_u_vol: float = 0.5                     # unstressed volume of the maternal placenta (L)
        self.plm_el_base: float= 5000.0                 # elastance of the maternal placenta (mmHg/L)
        self.dif_o2: float = 0.01                       # diffusion constant of oxygen (mmol/mmHg)
        self.dif_co2: float = 0.01                      # diffusion constant of carbon dioxide (mmol/mmHg)
        self.mat_to2: float = 6.5                       # maternal total oxygen concentration (mmol/L)
        self.mat_tco2: float = 23.0                     # maternal total carbin dioxide concentration (mmol/L)
        self.umb_clamped: bool = True                   # flags whether the umbilical vessels are clamped or not

        # -----------------------------------------------
        # initialize dependent parameters
        self.umb_art_flow: float = 0.0                  # flow in the umbilical artery (L/s)
        self.umb_art_velocity: float = 0.0              # velocity in the umbilical artery (m/s)
        self.umb_ven_flow: float = 0.0                  # flow in the umbilical vein (L/s)
        self.umb_ven_velocity: float = 0.0              # velocity in the umbilical vein (m/s)
        self.mat_po2: float = 0.0                       # maternal placenta oxygen partial pressure (mmHg)
        self.mat_pco2: float = 0.0                      # maternal placenta carbon dioxide partial pressure (mmHg)

        # -----------------------------------------------
        # local parameters
        self._umb_art: object = None                    # reference to the umbilical artery (BloodResistor)
        self._umb_ven: object = None                    # reference to the umbilical vein (BloodResistor)
        self._plm: object = None                        # reference to the fetal placenta (BloodCapacitance)
        self._plf: object = None                        # reference to the maternal placenta (BloodCapacitance)
        self._pl_gasex: object = None                   # reference to the gas exchanger between fetal and maternal placenta (GasExchanger)
        self._calc_blood_composition: object = None     # reference to the function of the Blood model which calculates the blood composition in a blood containg model
        self._placenta_parts: list = []                 # list holding all placental parts
        self._update_interval: float = 0.015            # update interval of the placenta model (s)
        self._update_counter: float = 0.0               # counter of the update interval (s)

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # get a reference to all ecls components for performance reasons
        self._umb_art = self._model_engine.models["UMB_ART"]
        self._umb_ven = self._model_engine.models["UMB_VEN"]
        self._plf = self._model_engine.models["PLF"]
        self._plm = self._model_engine.models["PLM"]
        self._pl_gasex = self._model_engine.models["PL_GASEX"]

        # clear the placenta part list
        self._placenta_parts = []
        # add the placenta parts to the list
        self._placenta_parts.append(self._umb_art)
        self._placenta_parts.append(self._umb_ven)
        self._placenta_parts.append(self._plf)
        self._placenta_parts.append(self._plm)
        self._placenta_parts.append(self._pl_gasex)

        # get a reference to the blood composition routine
        self._calc_blood_composition = self._model_engine.models["Blood"].calc_blood_composition

        # prepare placenta
        self.set_umb_art_diameter(self.umb_art_diameter)
        self.set_umb_art_length(self.umb_art_length)
        self.set_umb_ven_diameter(self.umb_ven_diameter)
        self.set_umb_ven_length(self.umb_ven_length)
        self.set_fetal_placenta_volume(self.plf_u_vol)
        self.set_fetal_placenta_elastance(self.plf_el_base)
        self.set_maternal_placenta_volume(self.plm_u_vol)
        self.set_maternal_placenta_elastance(self.plm_el_base)
        self.set_maternal_to2(self.mat_to2)
        self.set_maternal_tco2(self.mat_tco2)
        self.set_dif_o2(self.dif_o2)
        self.set_dif_co2(self.dif_co2)

        # clamp the umbilical cord
        self.clamp_umbilical_cord(True)

        # flag that the model is initialized
        self._is_initialized = True
        self._update_interval = 0.015
        self._update_counter = 0.0


    def calc_model(self) -> None:
        # increase the update counter
        self._update_counter += self._t
        if self._update_counter > self._update_interval:
            self._update_counter = 0.0

            # store the maternal p2o and pco2
            self.mat_po2 = self._plm.po2
            self.mat_pco2 = self._plm.pco2

            # store the arterial and venous flows
            self.umb_art_flow = self._umb_art.flow * 60.0
            self.umb_ven_flow = self._umb_ven.flow * 60.0

            # determine the area of the umb artery and veins depending on the diameter
            ua_area = math.pow((self.umb_art_diameter * 0.001) / 2.0, 2.0) * math.pi # in m^2
            uv_area = math.pow((self.umb_ven_diameter * 0.001) / 2.0, 2.0) * math.pi # in m^2

            # calculate the velocity = flow_rate (in m^3/s) / (pi * radius^2) in m/s
            if ua_area > 0:
                self.umb_art_velocity = ((self.umb_art_flow * 0.001) / ua_area) * 1.4
            if uv_area > 0:
                self.umb_ven_velocity = ((self.umb_ven_flow * 0.001) / uv_area) * 1.4

    def switch_placenta(self, state) -> None:
        # switch the calculations of the placenta model
        self.is_enabled = state

        # enable of disable all placenta components
        for pp in self._placenta_parts:
            pp.is_enabled = state
            # make sure the no flow flag is set correctly on the resistors (safety)
            if hasattr(pp, "no_flow"):
                pp.no_flow = not state

    def clamp_umbilical_cord(self, state) -> None:
        # determines whether or not the umbilical vessels are clamped or not by setting the no_flow property
        # of the umb vessels
        self.umb_clamped = state
        self._umb_art.no_flow = state
        self._umb_ven.no_flow = state

    def set_umb_art_diameter(self, new_diameter) -> None:
        self.umb_art_diameter = new_diameter
        # calculate the resistance
        self.umb_art_res = self.calc_tube_resistance(self.umb_art_diameter, self.umb_art_length)
        # reset the umbilical artery resistance
        self.set_umb_art_resistance(self.umb_art_res)

    def set_umb_art_length(self, new_length) -> None:
        self.umb_art_length = new_length
        # calculate the resistance
        self.umb_art_res = self.calc_tube_resistance(self.umb_art_diameter, self.umb_art_length)
        # reset the umbilical artery resistance
        self.set_umb_art_resistance(self.umb_art_res)

    def set_umb_art_resistance(self, new_res) -> None:
        # reset the umbilical artery resistance
        self.umb_art_res = new_res
        self._umb_art.r_for = self.umb_art_res
        self._umb_art.r_back = self._umb_art.r_for

    def set_umb_ven_diameter(self, new_diameter) -> None:
        self.umb_ven_diameter = new_diameter
        self.umb_ven_res = self.calc_tube_resistance(self.umb_ven_diameter, self.umb_ven_length)
        self.set_umb_ven_resistance(self.umb_ven_res)

    def set_umb_ven_length(self, new_length) -> None:
        self.umb_ven_length = new_length
        self.umb_ven_res = self.calc_tube_resistance(self.umb_ven_diameter, self.umb_ven_length)
        self.set_umb_ven_resistance(self.umb_ven_res)

    def set_umb_ven_resistance(self, new_res) -> None:
        # reset the umbilical vein resistance
        self.umb_ven_res = new_res
        self._umb_ven.r_for = self.umb_ven_res
        self._umb_ven.r_back = self._umb_art.r_for

    def set_fetal_placenta_volume(self, new_volume) -> None:
        self._plf.u_vol = new_volume
        self._plf.vol = new_volume
    
    def set_fetal_placenta_elastance(self, new_elastance) -> None:
        self._plf.el_base = new_elastance

    def set_maternal_to2(self, new_to2) -> None:
        self._plm.to2 = new_to2

    def set_maternal_tco2(self, new_tco2) -> None:
        self._plm.tco2 = new_tco2
    
    def set_maternal_solutes(self, new_solutes) -> None:
        for key, value in new_solutes.items():
            self._plm.solutes[key] = value

    def set_maternal_placenta_volume(self, new_volume) -> None:
        self._plm.u_vol = new_volume
        self._plm.vol = new_volume

    def set_maternal_placenta_elastance(self, new_elastance) -> None:
        self._plm.el_base = new_elastance

    def set_dif_o2(self, new_dif_o2) -> None:
        self.dif_o2 = new_dif_o2
        self._pl_gasex.dif_o2 = self.dif_o2
    
    def set_dif_co2(self, new_dif_co2) -> None:
        self.dif_co2 = new_dif_co2
        self._pl_gasex.dif_co2 = self.dif_co2

    def calc_tube_volume(self, diameter, length) -> float:
        # return the volume in liters
        return math.pi * math.pow(0.5 * diameter, 2) * length * 1000.0
    
    def calc_tube_resistance(self, diameter, length, viscosity=6.0) -> float:
        # resistance is calculated using Poiseuille's Law : R = (8 * n * L) / (PI * r^4)

        # we have to watch the units carefully where we have to make sure that the units in the formula are
        # resistance is in mmHg * s / l
        # L = length in meters
        # r = radius in meters
        # n = viscosity in centiPoise

        # convert viscosity from centiPoise to Pa * s
        n_pas = viscosity / 1000.0

        # convert the length to meters
        length_meters = length

        # calculate radius in meters
        radius_meters = diameter / 2

        # calculate the resistance    Pa *  / m3
        res = (8.0 * n_pas * length_meters) / (math.pi * math.pow(radius_meters, 4))

        # convert resistance of Pa/m3 to mmHg/l
        res = res * 0.00000750062
        return res

class Resuscitation(BaseModelClass):
    '''
    The Resuscitation class models a resuscitation situation where chest compressions and ventilations are
    performed at various different rates. 
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties
        self.cpr_enabled: bool = False                  # determines whether cpr is enabled or not
        self.chest_comp_freq: float = 100.0             # chest compressions frequency (compressions / min)
        self.chest_comp_max_pres: float = 10.0          # maximal pressure of the chest compressions (mmHg)
        self.chest_comp_targets: dict = { "THORAX": 0.1}# dictionary holding the target models of the chest compressions and the relative force
        self.chest_comp_no: int = 15                    # number of compressions if not continuous
        self.chest_comp_cont: bool = False              # determines whether the chest compressions are continuous
        
        self.vent_freq: float = 30.0                    # ventilations frequency (breaths / min)
        self.vent_no: int = 2                           # number of ventilatins if not continuous
        self.vent_pres_pip: float = 16.0                # peak pressure of the ventilations (cmH2O)
        self.vent_pres_peep: float = 5.0                # positive end expiratory pressure of the ventilations (cmH2O)
        self.vent_insp_time: float = 1.0                # inspiration time of the ventilations (s)
        self.vent_fio2: float = 0.21                    # fio2 of the inspired air

        # -----------------------------------------------
        # initialize dependent properties
        self.chest_comp_pres: float = 0.0               # compression pressure (mmHg)

        # -----------------------------------------------
        # local variables
        self._ventilator: object = None                 # reference to the mechanical ventilator model
        self._breathing: object = None                  # reference to the breathing model
        self._comp_timer: float = 0.0                   # compressions timer (s)
        self._comp_counter: int = 0.0                   # counter of the number of compressions
        self._comp_pause: bool = False                  # determines whether the compressions are paused or not
        self._comp_pause_interval: float = 2.0          # interval of the compressions pause (s)
        self._comp_pause_counter: float = 0.0           # compressions pause counter (s)
        self._vent_interval: float = 0.0                # interval between ventilations (s)
        self._vent_counter: float = 0.0                 # ventilation interval counter (s)

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # get references to the model on which this model depends
        self._ventilator = self._model_engine.models["Ventilator"]
        self._breathing = self._model_engine.models["Breathing"]
        
        # set the fio2 on the ventilator
        self.set_fio2(self.vent_fio2)

        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self) -> None:
        # return if cpr is not enabled
        if not self.cpr_enabled:
            return
        
        # calculate the compression pause (no of ventilations * breath duration)
        self._comp_pause_interval = (60.0 / self.vent_freq) * self.vent_no
        self._vent_interval = (self._comp_pause_interval / self.vent_no) + self._t
        # if the compressions are continuous set the ventilator frequency on the ventilator model
        if self.chest_comp_cont:
            self._ventilator.vent_rate = self.vent_freq
        else:
            # if the compressions or not continuous set a extremely low ventilator rate
            # as the ventilator breaths are triggered differently
            self._ventilator.vent_rate = 1.0


        # if there is a pause in the compressions
        if self._comp_pause:
            # increase the compressions pause counter
            self._comp_pause_counter += self._t

            # check whether the compression pause is over
            if self._comp_pause_counter > self._comp_pause_interval:
                self._comp_pause = False
                self._comp_pause_counter = 0.0
                self._comp_counter = 0
                self._vent_counter = 0.0
            
            # increase the ventilator interval timer
            self._vent_counter += self._t

            # trigger a breath when the ventilator counter is reached
            if self._vent_counter > self._vent_interval:
                self._vent_counter = 0.0
                self._ventilator.trigger_breath()
        else:
            # calculate the compression force according to y(t) = A sin(2PIft+o) where A = amplitude, f = frequency in Hz, t is time, o = phase shift
            a = self.chest_comp_max_pres / 2.0
            f = self.chest_comp_freq / 60.0
            self.chest_comp_pres = a * math.sin(2 * math.pi * f * self._comp_timer - 0.5 * math.pi) + a
            
            # increase the compressions timer
            self._comp_timer += self._t

            # check whether the compression has ended
            if self._comp_timer > (60.0 / self.chest_comp_freq):
                # reset the compression timer
                self._comp_timer = 0.0
                # increase the number of compressions
                self._comp_counter += 1

        # check whether we need to pause the compressions
        if self._comp_counter >= self.chest_comp_no and not self.chest_comp_cont:
            self._comp_pause = True
            self._comp_pause_counter = 0.0
            self._comp_counter = 0
            self._ventilator.trigger_breath()

        # apply the compression force to the target
        for key, value in self.chest_comp_targets.items():
            self._model_engine.models[key].pres_cc = float(self.chest_comp_pres * value)


    
    def switch_cpr(self, state):
        if state:
            self._ventilator.switch_ventilator(True)
            self._ventilator.set_pc(self.vent_pres_pip, self.vent_pres_peep, 1.0, self.vent_insp_time, 5.0)
            self._breathing.switch_breathing(False)
            self.cpr_enabled = True
        else:
            self.cpr_enabled = False

    def set_fio2(self, new_fio2):
        self._ventilator.set_fio2(new_fio2)

#----------------------------------------------------------------------------------------------------------------------------
class Circulation(BaseModelClass):
    '''
    The Circulation class is not a model but houses methods that influence groups of models. In case
    of the circulation class these groups contain models having to do with the blood circulation.
    E.g. the method change_systemic_vascular_resistance influences the systemic vascular resistance 
    by setting the r_for_factor, r_back_factor and  the el_base_factor of de BloodResistors and BloodCapacitances 
    stored in a list called svr_targets.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # independent properties
        self.syst_art_res_factor: float = 1.0           # systemic arteries resistance factor
        self.syst_ven_res_factor: float = 1.0           # systemic veins resistance factor
        self.syst_art_el_factor: float = 1.0            # systemic arteries elastance factor
        self.syst_ven_el_factor: float = 1.0            # systemic veins elastance factor
        self.syst_art_u_vol_factor: float = 1.0         # systemic arteries unstressed volume factor
        self.syst_ven_u_vol_factor: float = 1.0         # systemic veins unstressed volume factor

        self.pulm_art_res_factor: float = 1.0           # pulmonary arteries resistance factor
        self.pulm_ven_res_factor: float = 1.0           # pulmonary veins resistance factor
        self.pulm_art_el_factor: float = 1.0            # pulmonary arteries elastance factor
        self.pulm_ven_el_factor: float = 1.0            # pulmonary veins elastance factor
        self.pulm_art_u_vol_factor: float = 1.0         # pulmonary arteries unstressed volume factor
        self.pulm_ven_u_vol_factor: float = 1.0         # pulmonary veins unstressed volume factor

        self.syst_art_res_ans_factor: float = 1.0       # systemic arteries resistance ans factor
        self.syst_ven_res_ans_factor: float = 1.0       # systemic veins resistance ans factor
        self.syst_art_el_ans_factor: float = 1.0        # systemic arteries elastance ans factor
        self.syst_ven_el_ans_factor: float = 1.0        # systemic veins elastance ans factor
        self.syst_art_u_vol_ans_factor: float = 1.0     # systemic arteries unstressed volume ans factor
        self.syst_ven_u_vol_ans_factor: float = 1.0     # systemic veins unstressed volume ans factor

        self.pulm_art_res_ans_factor: float = 1.0       # pulmonary arteries resistance ans factor
        self.pulm_ven_res_ans_factor: float = 1.0       # pulmonary veins resistance ans factor
        self.pulm_art_el_ans_factor: float = 1.0        # pulmonary arteries elastance ans factor
        self.pulm_ven_el_ans_factor: float = 1.0        # pulmonary veins elastance ans factor
        self.pulm_art_u_vol_ans_factor: float = 1.0     # pulmonary arteries unstressed volume ans factor
        self.pulm_ven_u_vol_ans_factor: float = 1.0     # pulmonary veins unstressed volume ans factor

        # -----------------------------------------------
        # dependent properties
        self.total_blood_volume: float = 0.0            # holds the current total blood volume

        # -----------------------------------------------
        # local properties
        self._blood_containing_modeltypes: list = ["BloodCapacitance", "BloodTimeVaryingElastance"]
        self._update_interval: float = 0.015            # update interval (s)
        self._update_counter: float = 0.0               # update interval counter (s)

    def calc_model(self) -> None:
        self._update_counter += self._t
        if self._update_counter > self._update_interval:
            self._update_counter = 0.0

            # update the ans factors, as they are continuously set we need to update them
            for t in self._model_engine.model_groups["syst_arteries"]:
                t.el_base_ans_factor = self.syst_art_el_ans_factor
                t.u_vol_ans_factor = self.syst_art_u_vol_ans_factor

            for t in self._model_engine.model_groups["syst_art_resistors"]:
                t.r_ans_factor = self.syst_art_res_ans_factor

            for t in self._model_engine.model_groups["syst_veins"]:
                t.el_base_ans_factor = self.syst_ven_el_ans_factor
                t.u_vol_ans_factor = self.syst_ven_u_vol_ans_factor

            for t in self._model_engine.model_groups["syst_ven_resistors"]:
                t.r_ans_factor = self.syst_ven_res_ans_factor

            for t in self._model_engine.model_groups["pulm_arteries"]:
                t.el_base_ans_factor = self.pulm_art_el_ans_factor
                t.u_vol_ans_factor = self.pulm_art_u_vol_ans_factor

            for t in self._model_engine.model_groups["pulm_art_resistors"]:
                t.r_ans_factor = self.pulm_art_res_ans_factor

            for t in self._model_engine.model_groups["pulm_veins"]:
                t.el_base_ans_factor = self.pulm_ven_el_ans_factor
                t.u_vol_ans_factor = self.pulm_ven_u_vol_ans_factor

            for t in self._model_engine.model_groups["pulm_ven_resistors"]:
                t.r_ans_factor = self.pulm_ven_res_ans_factor

    def change_pulm_art_elastance(self, change: float) -> None:
        if change > 0.0:
            self.pulm_art_el_factor = change
            for t in self._model_engine.model_groups["pulm_arteries"]:
                    t.el_base_ans_factor = self.pulm_art_el_factor
    
    def change_pulm_art_u_vol(self, change: float) -> None:
        if change > 0.0:
            self.pulm_art_u_vol_factor = change
            for t in self._model_engine.model_groups["pulm_arteries"]:
                    t.r_ans_factor = self.pulm_art_u_vol_factor

    def change_pulm_art_resistance(self, change: float) -> None:
        if change > 0.0:
            self.pulm_art_res_factor = change
            for t in self._model_engine.model_groups["pulm_art_resistors"]:
                    t.r_factor = self.pulm_art_res_factor
             
    def change_pulm_ven_elastance(self, change: float) -> None:
        if change > 0.0:
            self.pulm_ven_el_factor = change
            for t in self._model_engine.model_groups["pulm_veins"]:
                    t.el_base_ans_factor = self.pulm_ven_el_factor
    
    def change_pulm_ven_u_vol(self, change: float) -> None:
        if change > 0.0:
            self.pulm_ven_u_vol_factor = change
            for t in self._model_engine.model_groups["pulm_veins"]:
                    t.r_ans_factor = self.pulm_ven_u_vol_factor

    def change_pulm_ven_resistance(self, change: float) -> None:
        if change > 0.0:
            self.pulm_ven_res_factor = change
            for t in self._model_engine.model_groups["pulm_ven_resistors"]:
                    t.r_factor = self.pulm_ven_res_factor
    
    def change_syst_art_elastance(self, change: float) -> None:
        if change > 0.0:
            self.syst_art_el_factor = change
            for t in self._model_engine.model_groups["syst_arteries"]:
                    t.el_base_ans_factor = self.syst_art_el_factor
    
    def change_syst_art_u_vol(self, change: float) -> None:
        if change > 0.0:
            self.syst_art_u_vol_factor = change
            for t in self._model_engine.model_groups["syst_arteries"]:
                    t.r_ans_factor = self.syst_art_u_vol_factor

    def change_syst_art_resistance(self, change: float) -> None:
        if change > 0.0:
            self.syst_art_res_factor = change
            for t in self._model_engine.model_groups["syst_art_resistors"]:
                    t.r_factor = self.syst_art_res_factor
             
    def change_syst_ven_elastance(self, change: float) -> None:
        if change > 0.0:
            self.syst_ven_el_factor = change
            for t in self._model_engine.model_groups["syst_veins"]:
                    t.el_base_ans_factor = self.syst_ven_el_factor
    
    def change_syst_ven_u_vol(self, change: float) -> None:
        if change > 0.0:
            self.syst_ven_u_vol_factor = change
            for t in self._model_engine.model_groups["syst_veins"]:
                    t.r_ans_factor = self.syst_ven_u_vol_factor

    def change_syst_ven_resistance(self, change: float) -> None:
        if change > 0.0:
            self.syst_ven_res_factor = change
            for t in self._model_engine.model_groups["syst_ven_resistors"]:
                    t.r_factor = self.syst_ven_res_factor

    def get_total_blood_volume(self) -> float:
        total_volume: float = 0.0
        # iterate over all blood containing models
        for _, m in self._model_engine.models.items():
            if (m.model_type in self._blood_containing_modeltypes):
                # if the model is enabled then at the current volume to the total volume
                if (m.is_enabled):
                    total_volume += m.vol

        # store the current volume
        self.total_blood_volume = total_volume

        # return the total blood volume
        return self.total_blood_volume

    def set_total_blood_volume(self, new_blood_volume: float) -> None:
        # first get the current volume
        current_blood_volume = self.get_total_blood_volume()
        # calculate the change in total blood volume
        blood_volume_change = new_blood_volume / current_blood_volume
        # iterate over all blood containing models
        for _, m in self._model_engine.models.items():
            if (m.model_type in self._blood_containing_modeltypes):
                if (m.is_enabled):
                    # change the volume with the blood_volume_change_factor
                    self._model_engine.models[m].vol = (self._model_engine.models[m].vol * blood_volume_change)
                    # also change the unstressed volume so that the ratio between volume and unstressed volume does not change!
                    self._model_engine.models[m].u_vol = (self._model_engine.models[m].u_vol * blood_volume_change)
        
        # store the new total blood volume
        self.total_blood_volume = self.get_total_blood_volume()

class Respiration(BaseModelClass):
    '''
    The Respiration class is not a model but houses methods that influence groups of models. In case
    of the respiration class these groups contain models having to do with the respiratory tract.
    E.g. the method change_lower_airway_resistance influences the resistance of the lower airways 
    by setting the r_factor of the DS_ALL and DS_ALR gas resistors stored in a list called lower_airways.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties
        self.dif_o2_factor: float = 1.0                 # o2 diffusion constant factor
        self.dif_co2_factor: float = 1.0                # co2 diffusion constant factor
        self.dead_space_u_vol_factor: float = 1.0       # dead space unstressed volume factor
        self.lung_el_factor: float = 1.0                # lungs elastance factor
        self.chestwall_el_factor: float = 1.0           # chestwall elastance factor
        self.thorax_el_factor: float = 1.0              # thorax elastance factor
        self.upper_aw_res_factor: float = 1.0           # upper airway resistance factor
        self.lower_aw_res_factor: float = 1.0           # lower airway resistance factor
        self.lung_shunt_res_factor: float = 1.0         # lung shunt resistance factor

        self.upper_airways: list = []                   # names of the upper airways
        self.dead_space: list = []                      # names of the dead spaces
        self.thorax: list = []                          # names of the thorax
        self.chestwall: list = []                       # names of the chestwalls
        self.alveolar_spaces: list = []                 # names of the alveolar spaces
        self.lower_airways: list = []                   # names of the lower airways
        self.gas_exchangers: list = []                  # names of the pulmonary gasexchangers
        self.lung_shunts: list = []                     # names of the lungshunts

        # -----------------------------------------------
        # initialize dependent properties

        # -----------------------------------------------
        # local properties
        self._upper_airways: list = []                  # references to the upper airways
        self._dead_space: list = []                     # references to the dead space
        self._thorax: list = []                         # references to the thorax
        self._chestwall: list = []                      # references to the chestwall
        self._alveolar_spaces: list = []                # references to the alveolar spaces
        self._lower_airways: list = []                  # references to the lower airways
        self._gas_exchangers: list = []                 # references to the gasexchangers
        self._lung_shunts: list = []                    # references to the lungshunts
        self._update_interval: float = 0.015            # update interval (s)
        self._update_counter: float = 0.0               # update interval counter (s)

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # store all reference
        for t in self.upper_airways:
            self._upper_airways.append(self._model_engine.models[t])
        for t in self.dead_space:
            self._dead_space.append(self._model_engine.models[t])
        for t in self.thorax:
            self._thorax.append(self._model_engine.models[t])
        for t in self._chestwall:
            self._chestwall.append(self._model_engine.models[t])
        for t in self.alveolar_spaces:
            self._alveolar_spaces.append(self._model_engine.models[t])
        for t in self.lower_airways:
            self._lower_airways.append(self._model_engine.models[t])
        for t in self.gas_exchangers:
            self._gas_exchangers.append(self._model_engine.models[t])
        for t in self.lung_shunts:
            self._lung_shunts.append(self._model_engine.models[t])

        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self) -> None:
        pass

    def change_intrapulmonary_shunting(self, change: float) -> None:
        if change > 0.0:
            self.lung_shunt_res_factor = change
            for t in self._lung_shunts:
                t.r_factor = self.lung_shunt_res_factor

    def change_dead_space(self, change: float) -> None:
        if change > 0.0:
            self.dead_space_u_vol_factor = change
            for t in self._dead_space:
                t.u_vol_factor = self.dead_space_u_vol_factor
    
    def change_upper_airway_resistance(self, change: float) -> None:
        if change > 0.0:
            self.upper_aw_res_factor = change
            for t in self._upper_airways:
                t.r_factor = self.upper_aw_res_factor

    def change_lower_airway_resistance(self, change: float) -> None:
        if change > 0.0:
            self.lower_aw_res_factor = change
            for t in self._lower_airways:
                t.r_factor = self.lower_aw_res_factor

    def change_thoracic_elastance(self, change: float) -> None:
        if change > 0.0:
            self.thorax_el_factor = change
            for t in self._thorax:
                t.el_base_factor = self.thorax_el_factor

    def change_lung_elastance(self, change: float) -> None:
        if change > 0.0:
            self.lung_el_factor = change
            for t in self._alveolar_spaces:
                t.el_base_factor = self.lung_el_factor

    def change_chestwall_elastance(self, change: float) -> None:
        if change > 0.0:
            self.chestwall_el_factor = change
            for t in self._chestwall:
                t.el_base_factor = self.chestwall_el_factor

    def change_diffusion_capacity(self, change: float) -> None:
         if change > 0.0:
            self.dif_o2_change = change
            self.dif_co2_change = change
            for t in self._gas_exchangers:
                t.dif_o2_factor = self.dif_o2_change
                t.dif_co2_factor = self.dif_co2_change

class Shunts(BaseModelClass):
    '''
    The Shunts class calculates the resistances of the shunts (ductus arteriosus, foramen ovale and ventricular septal defect) from the diameter and length
    It sets the resistances on the correct models reprensenting the Shunts.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties
        self.fo_enabled: bool = False                   # boolean determining whether the foramen ovale is enabled or not
        self.fo_diameter: float = 0.0                   # diameter of the foramen ovale (mm)
        self.fo_length: float = 2.0                     # thickness of the foramen ovale (mm)
        self.fo_backflow_factor: float = 1.0            # backflow resistance factor of the foramen ovale
        self.fo_r_k: float = 1.0                        # non linear resistance of the foramen ovale

        self.vsd_enabled: bool = False                  # boolean determining whether the ventricular septal defect is enabled or not
        self.vsd_diameter: float = 0.0                  # diameter of the ventricular septal defect (mm)       
        self.vsd_length: float = 2.0                    # thickness of the ventricular septal defect (mm)
        self.vsd_backflow_factor: float = 1.0           # backflow resistance factor of the ventricular septal defect
        self.vsd_r_k: float = 1.0                       # non linear resistance of the ventricular septal defect

        self.da_enabled: bool = False                   # boolean determining whether the ductus arteriosus is enabled or not
        self.da_el: float = 5000.0                      # elastance of the ductus arteriosus (mmHg / L)
        self.da_diameter: float = 0.0                   # diameter of the ductus arteriosus (mm)
        self.da_length: float = 10.0                    # length of the ductus arteriosus (mm)
        self.da_backflow_factor: float = 1.0            # backflow resistance factor of the ductus arteriosus
        self.da_r_k: float = 1.0                        # non linear resistance of the ductus arteriosus

        self.da: str = ""                               # name of the ductus arteriosus blood capacitance model
        self.da_in: str = ""                            # name of the ductus arteriosus inflow blood resistor 
        self.da_out: str = ""                           # name of the ductus arteriosus outflow blood resistor 
        self.fo: str = ""                               # name of the foramen ovale blood resistor 
        self.vsd: str = ""                              # name of the ventricular septal defect blood resistor 

        # -----------------------------------------------
        # initialize dependent properties
        self.da_r_for: float = 1000.0                   # calculated forward resistance across the ductus arteriosus (mmHg * s / L)
        self.da_r_back: float = 1000.0                  # calculated backward resistance across the ductus arteriosus (mmHg * s / L)
        self.da_flow: float = 0.0                       # flow across the ductus arteriosus (L/min)
        self.da_velocity: float = 0.0                   # velocity of the flow across the ductus arteriosus (m/s)
        self.fo_r_for: float = 1000.0                   # calculated forward resistance across the foramen ovale (mmHg * s / L)
        self.fo_r_back: float = 1000.0                  # calculated forward resistance across the foramen ovale (mmHg * s / L)
        self.fo_flow: float = 0.0                       # flow across the foramen ovale (L/min)
        self.fo_velocity: float = 0.0                   # velocity of the flow across the foramen ovale (m/s)
        self.vsd_r_for: float = 1000.0                  # calculated forward resistance across the ventricular septal defect (mmHg * s / L)
        self.vsd_r_back: float = 1000.0                 # calculated forward resistance across the ventricular septal defect (mmHg * s / L)
        self.vsd_flow: float = 0.0                      # flow across the ventricular septal defect (L/min)
        self.vsd_velocity: float = 0.0                  # velocity of the flow across the ventricular septal defect (m/s)

        # -----------------------------------------------
        # local properties
        self._update_interval = 0.015                   # update interval (s)
        self._update_counter = 0.0                      # update interval counter (s)
        self._da = None                                 # reference to the ductus arteriosus blood capacitance
        self._da_in = None                              # reference to the ductus arteriosus inflow resistance
        self._da_out = None                             # reference to the ductus arteriosus outflow resistance
        self._fo = None                                 # reference to the foramen ovale blood resistor
        self._vsd = None                                # reference to the ventricular septal defect resistor
        self._da_area: float = 0.0                      # area of the ductus arteriosus (m^2)
        self._fo_area: float = 0.0                      # area of the foramen ovale (m^2)
        self._vsd_area: float = 0.0                     # area of the ventricular septal defect (m^2)
        

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # store the references to the models for performance reasons
        self._da = self._model_engine.models[self.da]
        self._da_in = self._model_engine.models[self.da_in]
        self._da_out = self._model_engine.models[self.da_out]
        self._fo = self._model_engine.models[self.fo]
        self._vsd = self._model_engine.models[self.vsd]

        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self) -> None:
        if self._update_counter > self._update_interval:
            self._update_counter = 0.0

            # update the diameters and other properties
            self.set_ductus_arteriosus_properties(self.da_diameter, self.da_length)
            self.set_foramen_ovale_properties(self.fo_diameter, self.fo_length)
            self.set_ventricular_septal_defect_properties(self.vsd_diameter, self.vsd_length)

        self._update_counter += self._t

        # store the flows and velocities
        self.da_flow = self._da_out.flow * 60.0
        self.fo_flow = self._fo.flow * 60.0
        self.vsd_flow = self._vsd.flow * 60.0

        # calculate the velocity = flow_rate (in m^3/s) / (pi * radius^2) in m/s
        if self._da_area > 0:
            self.da_velocity = ((self.da_flow * 0.001) / self._da_area) * 1.4
        if self._fo_area > 0:
            self.fo_velocity = ((self.fo_flow * 0.001) / self._fo_area) * 1.4
        if self._vsd_area > 0:
            self.vsd_velocity = ((self.vsd_flow * 0.001) / self._vsd_area) * 1.4

    def set_ductus_arteriosus_properties(self, new_diameter: float, new_length: float) -> None:
        if new_diameter and self.da_enabled > 0.0:
            self.da_diameter = new_diameter
            self.da_length = new_length
            self._da_area = math.pow((self.da_diameter * 0.001) / 2.0, 2.0) * math.pi # in m^2
            self.da_r_for = self.calc_resistance(self.da_diameter, self.da_length)
            self.da_r_back = self.da_r_for * self.da_backflow_factor
            self._da_out.r_for = self.da_r_for
            self._da_out.r_back = self.da_r_back
            self._da.el_base = self.da_el
            self._da.is_enabled = True
            self._da_in.is_enabled = True
            self._da_out.is_enabled = True
            self._da_in.no_flow = False
            self._da_out.no_flow = False
        else:
            self.da_diameter = 0.0
            self._da_out.no_flow = True
            self._da_in.no_flow = True

    def set_foramen_ovale_properties(self, new_diameter: float, new_length: float) -> None:
        if new_diameter and self.fo_enabled > 0.0:
            self.fo_diameter = new_diameter
            self.fo_length = new_length
            self._fo_area = math.pow((self.fo_diameter * 0.001) / 2.0, 2.0) * math.pi # in m^2
            self.fo_r_for = self.calc_resistance(self.fo_diameter, self.fo_length)
            self.fo_r_back = self.fo_r_for * self.fo_backflow_factor
            self._fo.is_enabled = True
            self._fo.no_flow = False
        else:
            self.fo_diameter = 0.0
            self._fo.no_flow = True

    def set_ventricular_septal_defect_properties(self, new_diameter: float, new_length: float) -> None:
        if new_diameter and self.vsd_enabled > 0.0:
            self.vsd_diameter = new_diameter
            self.vsd_length = new_length
            self._vsd_area = math.pow((self.vsd_diameter * 0.001) / 2.0, 2.0) * math.pi # in m^2
            self.vsd_r_for = self.calc_resistance(self.vsd_diameter, self.vsd_length)
            self.vsd_r_back = self.fo_r_for * self.vsd_backflow_factor
            self._vsd.is_enabled = True
            self._vsd.no_flow = False
        else:
            self.vsd_diameter = 0.0
            self._vsd.no_flow = True

    def calc_resistance(self, diameter: float, length: float = 2.0, viscosity=6.0):
        if diameter > 0.0 and length > 0.0:
            # resistance is calculated using Poiseuille's Law : R = (8 * n * L) / (PI * r^4)
            # diameter (mm), length (mm), viscosity (cP)

            # convert viscosity from centiPoise to Pa * s
            n_pas = viscosity / 1000.0

            # convert the length to meters
            length_meters = length / 1000.0

            # calculate radius in meters
            radius_meters = diameter / 2 / 1000.0

            # calculate the resistance    Pa * s / m3
            res = (8.0 * n_pas * length_meters) / (math.pi * math.pow(radius_meters, 4))

            # convert resistance of Pa * s / m3 to mmHg * s/l
            res = res * 0.00000750062
            return res
        else:
            return 100000000

class Monitor(BaseModelClass):
    '''
    The Monitor class models a patient monitor and samples vital parameters
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties
        self.hr_avg_beats: float = 5.0                  # the number of beats for averaging the heartrate
        self.rr_avg_time: float = 20.0                  # averaging time of the respiratory rate
        self.sat_avg_time: float = 5.0                  # averaging time of the pulse oximeter
        self.sat_sampling_interval: float = 1.0         # sample interval of the pulse oximeter (s)

        # measuring sites mapping to models
        self.heart: str = "Heart"                       # name of the heart model
        self.ascending_aorta: str = "AA"                # name of the ascending aorta model
        self.descending_aorta: str = "AD"               # name of the descending aorta model
        self.pulm_artery: str = "PA"
        self.right_atrium: str = "RA"
        self.breathing: str = "Breathing"
        self.ventilator: str = "Ventilator"
        self.aortic_valve: str = "LV_AA"
        self.pulm_valve: str = "RV_PA"
        self.cor_ra: str = "COR_RA"
        self.aa_brain: str = "AA_BR"
        self.ad_kid: str = "AA_KID"
        self.ivc_ra: str = "IVCI_RA"
        self.svc_ra: str = "SVC_RA"
        self.thorax: str = "THORAX"
        self.deadspace: str = "DS"

        # -----------------------------------------------
        # initialize dependent properties

        self.heart_rate: float = 0.0                    # heartrate (bpm)
        self.resp_rate: float = 0.0                     # respiratory rate (/min)
        self.abp_syst: float = 0.0                      # arterial blood pressure systole (mmHg)
        self.abp_diast: float = 0.0                     # arterial blood pressure diastole (mmHg)
        self.abp_mean: float = 0.0                      # arterial blood pressure mean (mmHg)
        self.abp_pre_syst : float = 0.0                 # arterial blood pressure systole (mmHg)
        self.abp_pre_diast : float = 0.0                # arterial blood pressure diastole (mmHg)
        self.abp_pre_mean : float = 0.0                 # arterial blood pressure mean (mmHg)
        self.pap_syst : float = 0.0                     # pulmonary artery pressure systole (mmHg)
        self.pap_diast : float = 0.0                    # pulmonary artery pressure diastole (mmHg)
        self.pap_mean : float = 0.0                     # pulmonary artery pressure mean (mmHg)
        self.cvp : float = 0.0                          # central venous pressure (mmHg)
        self.spo2 : float = 0.0                         # arterial oxygen saturation in descending aorta (%)
        self.spo2_pre : float = 0.0                     # arterial oxygen saturation in ascending aorta (%)
        self.spo2_ven : float = 0.0                     # venous oxygen saturation in right atrium (%)
        self.etco2 : float = 0.0                        # end tidal partial pressure of carbon dioxide (kPa)
        self.temp : float = 0.0                         # blood temperature (dgs C)
        self.co : float = 0.0                           # cardiac output (l/min)
        self.ci : float = 0.0                           # cardiac index (l/min/m2)
        self.lvo : float = 0.0                          # left ventricular output (l/min)
        self.rvo : float = 0.0                          # right ventricular output (l/min)
        self.lv_sv : float = 0.0                        # left ventricular stroke volume (ml)
        self.rv_sv : float = 0.0                        # right ventricular stroke volume (ml)
        self.ivc_flow : float = 0.0                     # inferior vena cava flow (l/min)
        self.svc_flow : float = 0.0                     # superior vena cava flow (l/min)
        self.cor_flow : float = 0.0                     # coronary flow (l/min)
        self.brain_flow : float = 0.0                   # brain flow (l/min)
        self.kid_flow : float = 0.0                     # kidney flow (l/min)
        self.fio2 : float = 0.0                         # inspired fraction of oxygen
        self.pip : float = 0.0                          # peak inspiratory pressure (cmH2O)
        self.p_plat : float = 0.0                       # plateau inspiratory pressure (cmH2O)
        self.peep : float = 0.0                         # positive end expiratory pressure (cmH2O)
        self.tidal_volume : float = 0.0                 # tidal volume (l)
        self.ph : float = 0.0                           # arterial ph
        self.po2 : float = 0.0                          # arterial po2 (kPa)
        self.pco2 : float = 0.0                         # arterial pco2 (kPa)
        self.hco3 : float = 0.0                         # arterial bicarbonate concentration (mmol/l)
        self.be : float = 0.0                           # arterial base excess concentration (mmol/l)

        # signals
        self.ecg_signal : float = 0.0                   # ecg signal
        self.abp_signal : float = 0.0                   # abp signal
        self.pap_signal : float = 0.0                   # pap signal
        self.cvp_signal : float = 0.0                   # cvp signal
        self.spo2_pre_signal : float = 0.0              # pulse-oximeter signal
        self.spo2_signal : float = 0.0                  # pulse-oximeter signal
        self.resp_signal : float = 0.0                  # respiratory signal
        self.co2_signal : float = 0.0                   # co2 signal

        # -----------------------------------------------
        # initialize local properties

        self._heart = None                              # reference to the heart model
        self._breathing = None                          # reference to the breathing model
        self._ventilator = None                         # reference to the mechanical ventilator model
        self._aa = None                                 # reference to the ascending aorta
        self._ad = None                                 # reference to the descending aorta
        self._ra = None                                 # reference to the right atrium
        self._pa = None                                 # reference to the pulmonary artery
        self._ds = None                                 # reference to the upper airway deadspace
        self._thorax = None                             # reference to the thorax
        self._lv_aa = None                              # reference to the aortic valve
        self._rv_pa = None                              # reference to the pulmonary valve
        self._ivc_ra = None                             # reference to the inferior cava to right atrium connector
        self._svc_ra = None                             # reference to the superior cava to right atrium connector
        self._cor_ra = None                             # reference to the coronaries to right atrium connector
        self._aa_br = None                              # reference to the ascending aorta to brain connector
        self._ad_kid = None                             # reference to the descending aorta to kidneys connector

        self._temp_aa_pres_max = -1000.0
        self._temp_aa_pres_min = 1000.0
        self._temp_ad_pres_max = -1000.0
        self._temp_ad_pres_min = 1000.0
        self._temp_ra_pres_max = -1000.0
        self._temp_ra_pres_min = 1000.0
        self._temp_pa_pres_max = -1000.0
        self._temp_pa_pres_min = 1000.0
        self._lvo_counter = 0.0
        self._rvo_counter = 0.0
        self._cor_flow_counter = 0.0
        self._ivc_flow_counter = 0.0
        self._svc_flow_counter = 0.0
        self._brain_flow_counter = 0.0
        self._kid_flow_counter = 0.0
        self._hr_list = []
        self._rr_list = []
        self._spo2_list = []
        self._spo2_pre_list = []
        self._spo2_ven_list = []
        self._rr_avg_counter = 0.0
        self._sat_avg_counter = 0.0
        self._sat_sampling_counter = 0.0
        self._beats_counter = 0
        self._beats_time = 0.0


    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # get the references to the models (for performance reasons)
        self._heart = self._model_engine.models.get(self.heart, None)
        self._ra = self._model_engine.models.get(self.right_atrium, None)
        self._breathing = self._model_engine.models.get(self.breathing, None)
        self._ventilator = self._model_engine.models.get(self.ventilator, None)
        self._ds = self._model_engine.models.get(self.deadspace, None)
        self._thorax = self._model_engine.models.get(self.thorax, None)
        self._aa = self._model_engine.models.get(self.ascending_aorta, None)
        self._ad = self._model_engine.models.get(self.descending_aorta, None)
        self._pa = self._model_engine.models.get(self.pulm_artery, None)
        self._lv_aa = self._model_engine.models.get(self.aortic_valve, None)
        self._rv_pa = self._model_engine.models.get(self.pulm_valve, None)
        self._ivc_ra = self._model_engine.models.get(self.ivc_ra, None)
        self._svc_ra = self._model_engine.models.get(self.svc_ra, None)
        self._cor_ra = self._model_engine.models.get(self.cor_ra, None)
        self._aa_br = self._model_engine.models.get(self.aa_brain, None)
        self._ad_kid = self._model_engine.models.get(self.ad_kid, None)

        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self) -> None:
        # collect the pressures
        self.collect_pressures()

        # collect the flows
        self.collect_blood_flows()

        # collect the signals
        self.collect_signals()

        # collect temperature
        self.temp = self._aa.temp

        # collect end tidal co2
        self.etco2 = self._ventilator.etco2

        # Determine the beginning of the cardiac cycle
        if self._heart.ncc_ventricular == 1:
            # Heart rate determination
            self._hr_list.append(self._heart.heart_rate)
            self.heart_rate = sum(self._hr_list) / len(self._hr_list)

            if len(self._hr_list) > self.hr_avg_beats:
                self._hr_list.pop(0)

            # Add 1 beat
            self._beats_counter += 1

            # Blood pressures
            if self._aa:
                self.abp_pre_syst = self._temp_aa_pres_max
                self.abp_pre_diast = self._temp_aa_pres_min
                self.abp_pre_mean = (2 * self._temp_aa_pres_min + self._temp_aa_pres_max) / 3.0
                self._temp_aa_pres_max = -1000.0
                self._temp_aa_pres_min = 1000.0

            if self._ad:
                self.abp_syst = self._temp_ad_pres_max
                self.abp_diast = self._temp_ad_pres_min
                self.abp_mean = (2 * self._temp_ad_pres_min + self._temp_ad_pres_max) / 3.0
                self._temp_ad_pres_max = -1000.0
                self._temp_ad_pres_min = 1000.0

            if self._ra:
                self.cvp = (2 * self._temp_ra_pres_min + self._temp_ra_pres_max) / 3.0
                self._temp_ra_pres_max = -1000.0
                self._temp_ra_pres_min = 1000.0

            if self._pa:
                self.pap_syst = self._temp_pa_pres_max
                self.pap_diast = self._temp_pa_pres_min
                self.pap_mean = (2 * self._temp_pa_pres_min + self._temp_pa_pres_max) / 3.0
                self._temp_pa_pres_max = -1000.0
                self._temp_pa_pres_min = 1000.0

        # Cardiac outputs
        if self._beats_counter > self.hr_avg_beats:
            if self._lv_aa:
                self.lvo = (self._lvo_counter / self._beats_time) * 60.0
                self._lvo_counter = 0.0

            if self._rv_pa:
                self.rvo = (self._rvo_counter / self._beats_time) * 60.0
                self._rvo_counter = 0.0

            if self._ivc_ra:
                self.ivc_flow = (self._ivc_flow_counter / self._beats_time) * 60.0
                self._ivc_flow_counter = 0.0

            if self._svc_ra:
                self.svc_flow = (self._svc_flow_counter / self._beats_time) * 60.0
                self._svc_flow_counter = 0.0

            if self._cor_ra:
                self.cor_flow = (self._cor_flow_counter / self._beats_time) * 60.0
                self._cor_flow_counter = 0.0

            if self._aa_br:
                self.brain_flow = (self._brain_flow_counter / self._beats_time) * 60.0
                self._brain_flow_counter = 0.0

            if self._ad_kid:
                self.kid_flow = (self._kid_flow_counter / self._beats_time) * 60.0
                self._kid_flow_counter = 0.0

            # Reset the counters
            self._beats_counter = 0
            self._beats_time = 0.0

        self._beats_time += self._t

        # Respiratory rate (rolling average)
        self._rr_avg_counter += self._t
        if self._rr_avg_counter > self.rr_avg_time:
            self._rr_list.pop(0)

        if self._breathing.ncc_insp == 1:
            self._rr_list.append(self._breathing.resp_rate)
            self.resp_rate = sum(self._rr_list) / len(self._rr_list)

        # Saturation
        if self._sat_avg_counter > self.sat_avg_time:
            self._sat_avg_counter = 0.0
            self._spo2_list.pop(0)
            self._spo2_pre_list.pop(0)
            self._spo2_ven_list.pop(0)

        if self._sat_sampling_counter > self.sat_sampling_interval:
            self._sat_sampling_counter = 0.0
            self._spo2_list.append(self._ad.so2)
            self._spo2_pre_list.append(self._aa.so2)
            self._spo2_ven_list.append(self._ra.so2)

            self.spo2 = sum(self._spo2_list) / len(self._spo2_list)
            self.spo2_pre = sum(self._spo2_pre_list) / len(self._spo2_pre_list)
            self.spo2_ven = sum(self._spo2_ven_list) / len(self._spo2_ven_list)

        self._sat_avg_counter += self._t
        self._sat_sampling_counter += self._t


    def collect_pressures(self) -> None:
        self._temp_aa_pres_max = (max(self._temp_aa_pres_max, self._aa.pres_in) if self._aa else -1000)
        self._temp_aa_pres_min = (min(self._temp_aa_pres_min, self._aa.pres_in) if self._aa else 1000)
        self._temp_ad_pres_max = (max(self._temp_ad_pres_max, self._ad.pres_in) if self._ad else -1000)
        self._temp_ad_pres_min = (min(self._temp_ad_pres_min, self._ad.pres_in) if self._ad else 1000)
        self._temp_ra_pres_max = (max(self._temp_ra_pres_max, self._ra.pres_in) if self._ra else -1000)
        self._temp_ra_pres_min = (min(self._temp_ra_pres_min, self._ra.pres_in) if self._ra else 1000)
        self._temp_pa_pres_max = (max(self._temp_pa_pres_max, self._pa.pres_in) if self._pa else -1000)
        self._temp_pa_pres_min = (min(self._temp_pa_pres_min, self._pa.pres_in) if self._pa else 1000)

    def collect_blood_flows(self) -> None:
        self._lvo_counter += self._lv_aa.flow * self._t if self._lv_aa else 0.0
        self._rvo_counter += self._rv_pa.flow * self._t if self._rv_pa else 0.0
        self._cor_flow_counter += self._cor_ra.flow * self._t if self._cor_ra else 0.0
        self._ivc_flow_counter += self._ivc_ra.flow * self._t if self._ivc_ra else 0.0
        self._svc_flow_counter += self._svc_ra.flow * self._t if self._svc_ra else 0.0
        self._brain_flow_counter += self._aa_br.flow * self._t if self._aa_br else 0.0
        self._kid_flow_counter += self._ad_kid.flow * self._t if self._ad_kid else 0.0

    def collect_signals(self) -> None:
        self.ecg_signal = self._heart.ecg_signal if self._heart else 0.0
        self.resp_signal = self._thorax.vol if self._thorax else 0.0
        self.spo2_pre_signal = self._aa.pres_in if self._aa else 0.0
        self.spo2_signal = self._ad.pres_in if self._ad else 0.0
        self.abp_signal = self._ad.pres_in if self._ad else 0.0
        self.pap_signal = self._pa.pres_in if self._pa else 0.0
        self.cvp_signal = self._ra.pres_in if self._ra else 0.0
        self.co2_signal = self._ventilator.co2 if self._ventilator else 0.0


#----------------------------------------------------------------------------------------------------------------------------
# custom classes from explain users (not verified by the explain team)
class ExampleCustomModel(BaseModelClass):
    '''
    The BloodResistor model is a extension of the Resistor model as described in the paper.
    A BloodResistor model is a connector between two blood containing models (e.g. BloodCapacitance or BloodTimeVaryingElastance) and
    the model determines the flow between the two models it connects.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties

        # -----------------------------------------------
        # initialize dependent properties

        # -----------------------------------------------
        # local variables

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self) -> None:
        pass

class RenalAutoregulation(BaseModelClass):
    '''
    The BloodResistor model is a extension of the Resistor model as described in the paper.
    A BloodResistor model is a connector between two blood containing models (e.g. BloodCapacitance or BloodTimeVaryingElastance) and
    the model determines the flow between the two models it connects.
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class
        super().__init__(model_ref, name)

        # -----------------------------------------------
        # initialize independent properties
        self.delay_MR: float = 0.3
        self.delay_TGF: float =18
        self.delta1: float = 0.3
        self.delta2: float = 1.2
        self.k: float = 0.5
        self.p0: float = 80   
        self.p1: float = 180  
        self.q0: float = 0.014
        self.ps_test: float = 80
        self.tau1: float = 4  
        self.tau2: float = 5.3            
        self.op_GFR: float = 110
        self.th_GFR: float = 106
        self.sa_GFR: float = 246
        self.tau3: float = 15
        self.tau4: float = 33
        self.g_GFR: float = 200
        self.g_MR: float = 1


        # -----------------------------------------------
        # initialize dependent properties
        self.GFR: float = 0.0
        self.RBF: float = 0.0
        self.ps: float = 0.0
        self.pt_max: float = 0.0
        self.dR_MR: float = 0.0
        self.dR_TGF: float = 0.0
        self.test = 0

        # -----------------------------------------------
        # local variables
        self._counter_MR: int = 0 
        self._d_MR: float = 0.0
        self._counter_TGF: int = 0
        self._d_GFR: float = 0.0

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in args.items():
            setattr(self, key, value)

        # do some custom model initialization
        # Myogenic response
        self._window_MR: int = (self.delta2 - self.delta1)/self._t
        self._pt_array = [0]*(int(self._window_MR))
        self._ps_array = [0]*(int(self.delay_MR/self._t))  # delay_MR = 0.3/dt
        
        # Tubuloglomerular feedback
        self._window_TGF: int = (self.delay_TGF/self._t)
        self._GFR_array = [self.op_GFR]*(int(self._window_TGF))  
        self._a_GFR: float = 0.0
        
        # Resistance
        R_GL_TU_VR = self._model_engine.models['GL_TU'].r_for + self._model_engine.models['TU_VR'].r_for # parallel resistances 
        self.R_rest = 1/(1/self._model_engine.models['GL_VR'].r_for + 1/R_GL_TU_VR) + self._model_engine.models['VR_VC'].r_for # renal resistance without AR_GL
        self.R_base =  self._model_engine.models['AR_GL'].r_for # baseline resistance of afferent arteriole
        
        # flag that the model is initialized
        self._is_initialized = True

    def calc_model(self) -> None:
        self.GFR = self._model_engine.models['GL_TU'].flow * 60 * 1000 # GFR in ml/min
        self.RBF = self._model_engine.models['AR_GL'].flow * 60 * 1000 # GFR in ml/min
        R_aff = self._model_engine.models['AR_GL'].r_for 

        # Myogenic Response
        pt = self._model_engine.models['AR'].pres   # pressure in AR at time t
        pVC = self._model_engine.models['VC'].pres  # pressure in VC at time t

        self._pt_array.append(pt) 
        self._pt_array.pop(0)# Insert pt at the end and remove first element. 

        pt_max = max(self._pt_array)
        self._ps_array.append(pt_max)
        self._ps_array.pop(0)
      
        if (self._counter_MR <= (self.delay_MR/self._t)):
            self._counter_MR += 1
            self.ps = pt_max
        else: 
            self.ps = self._ps_array[0]
            self._counter_MR = 2*(self.delay_MR/self._t)
        
        if (self.ps >= self.p1):
            cT = (self.q0 / (self.p1 - pVC)) * (1+self.k*(self.p1-self.p0) / self.p0)  # upper limit
            RT = 1/cT - self.R_rest
        elif (self.ps <= self.p0):
            cT = self.q0 / (self.p0 - pVC)    # lower limit
            RT = 1/cT - self.R_rest
        else: 
            cT = (self.q0 / (self.ps - pVC)) * (1+self.k*(self.ps-self.p0)/self.p0) # between p0 en p1
            RT = 1/cT - self.R_rest

        _a_MR = RT - self.R_base
        
        if (R_aff > RT):
            self._d_MR = self._t * (1/self.tau2) * (-self._d_MR + _a_MR) + self._d_MR   # vasodilatation
        else: self._d_MR = self._t * (1/self.tau1) * (-self._d_MR + _a_MR) + self._d_MR # vasoconstriction  
        
        self.dR_MR = self._d_MR * self.g_MR 

        # Tubuloglomerular Feedback
        self._GFR_array.append(self.GFR)
        self._GFR_array.pop(0)
        
        # if (self._counter_TGF <= (2*self.delay_TGF/self._t)):
        #     self._counter_TGF += 1
        #     _GFR_effect = self.op_GFR
            
        # else: 
        #     _GFR_effect = self._GFR_array[0]
        #     self._counter_TGF = 3*(self.delay_TGF/self._t)

        _GFR_effect = self._GFR_array[0]
        self._counter_TGF = 3*(self.delay_TGF/self._t)
        
        a_pre =  self._a_GFR
        self._a_GFR = self.activation_function(_GFR_effect, self.sa_GFR, self.op_GFR, self.th_GFR)
        
        if (self._a_GFR > a_pre):
            self._d_GFR = self._t * ((1 / self.tau3) * (-self._d_GFR + self._a_GFR)) + self._d_GFR; # vasoconstriction
        else: 
            self._d_GFR = self._t * ((1 / self.tau4) * (-self._d_GFR + self._a_GFR)) + self._d_GFR; # vasodilatation 
            
        self.dR_TGF = self._d_GFR*self.g_GFR 
       
        R = self.R_base + self.dR_MR + self.dR_TGF
        
        self.test = _GFR_effect
        self._model_engine.models['AR_GL'].r_for = R
        self._model_engine.models['AR_GL'].r_back = R

    def activation_function(self, value, saturation, operating_point, threshold) -> float:
        activation = 0

        if value >= saturation:
            activation = saturation - operating_point
        else:
            if value <= threshold:
                activation = threshold - operating_point
            else:
                activation = value - operating_point

        return activation

#----------------------------------------------------------------------------------------------------------------------------
# helper classes, these classes instantiate, initialize and run the model, collect and plot the data and perform tasks (e.g. closing a resistor, scaling etc)
class Datacollector():
    def __init__(self, model: object) -> None:
        # store a reference to the model instance
        self.model = model

        # define the watch list
        self.watch_list = []

        # define the data sample interval
        self.sample_interval = 0.005
        self._interval_counter = 0

        # get the modeling stepsize from the model
        self.modeling_stepsize = self.model.modeling_stepsize

        # try to add two always needed ecg properties to the watchlist
        self.ncc_ventricular = {'label': 'Heart.ncc_ventricular', 'model': self.model.models['Heart'], 'prop1': 'ncc_ventricular', 'prop2': None}
        self.ncc_atrial = {'label': 'Heart.ncc_atrial', 'model': self.model.models['Heart'], 'prop1': 'ncc_atrial', 'prop2': None}

        # add the two always there
        self.watch_list.append(self.ncc_atrial)
        self.watch_list.append(self.ncc_ventricular)

        # define the data list
        self.collected_data = []
    
    def clear_data(self):
        self.collected_data = []

    def clear_watchlist(self):
        # first clear all data
        self.clear_data()

        # empty the watch list
        self.watch_list = []

        # add the two always there
        self.watch_list.append(self.ncc_atrial)
        self.watch_list.append(self.ncc_ventricular)

    def set_sample_interval(self, new_interval=0.005):
        self.sample_interval = new_interval

    def add_to_watchlist(self, properties) -> bool:
        # define an return object
        success = True

        # first clear all data
        self.clear_data()

        # check whether property is a string
        if isinstance(properties, str):
            # convert string to a list
            properties = [properties]

        # add to the watchlist
        for prop in properties:
            # check whether the property is already in the watchlist
            duplicate: bool = False
            for wl_item in self.watch_list:
                if wl_item['label'] == prop:
                    duplicate = True
                    break

            # if the property is not yet present then process it
            if not duplicate:
                # process the property as it has shape MODEL.prop1.prop2
                processed_prop = self.find_model_prop(prop)

                # check whether the property is found and if so, add it to the watchlist
                if processed_prop is not None:
                    self.watch_list.append(processed_prop)
                else:
                    success = False

        return success
    
    def find_model_prop(self, prop):
        # split the model from the prop
        t = prop.split(sep=".")

        # if only 1 property is present
        if (len(t) == 2):
            # try to find the parameter in the model
            if t[0] in self.model.models:
                if (hasattr(self.model.models[t[0]], t[1])):
                    r = getattr(self.model.models[t[0]], t[1])
                    return {'label': prop, 'model': self.model.models[t[0]], 'prop1': t[1], 'prop2': None, 'ref': r}

        # if 2 properties are present
        if (len(t) == 3):
            # try to find the parameter in the model
            if t[0] in self.model.models:
                if (hasattr(self.model.models[t[0]], t[1])):
                    return {'label': prop, 'model': self.model.models[t[0]], 'prop1': t[1], 'prop2': t[2]}

        return None
    
    def collect_data(self, model_clock: float) -> None:
        # collect data at specific intervals set by the sample_interval
        if self._interval_counter >= self.sample_interval:
            # reset the interval counter
            self._interval_counter = 0
            # declare a data object holding the current model time
            data_object: object = {'time': round(model_clock, 4)}
            # process the watch_list
            for parameter in self.watch_list:
                # get the value of the model variable as stated in the watchlist
                prop1 = parameter['prop1']
                prop2 = parameter.get('prop2')
                value = getattr(parameter['model'], prop1)
                # faster way maybe: value = parameter['model'].__dict__.get(prop1)
                if prop2 is not None:
                    value = value.get(prop2, 0)

                # at the value to the data object
                data_object[parameter['label']] = value
            # at the data object to the collected data list
            self.collected_data.append(data_object)
        # increase the interval counter
        self._interval_counter += self.modeling_stepsize

class TaskScheduler():
    def __init__(self, model_ref: object) -> None:
        self._model_engine: object = model_ref          # object holding a reference to the model engine
        self._t: float = model_ref.modeling_stepsize    # setting the modeling stepsize
        self._is_initialized: bool = False              # flag whether the model is initialized or not
        self.is_enabled: bool = True                    # fg
        # local properties
        self._tasks: dict = {}                          # dictionary holding the current tasks
        self._task_interval: float = 0.015              # interval at which task are evaluated
        self._task_interval_counter: float = 0.0        # counter

    def add_task(self, new_task):
        id = "task_" + str(new_task["id"])
        del new_task["id"]
        new_task["running"] = False
        new_task["completed"] = False

        current_value = getattr(new_task["model"], new_task["prop1"])
        if new_task["prop2"] is not None:
            current_value = getattr(current_value, new_task["prop2"])
        new_task["current_value"] = current_value

        if isinstance(current_value, float) or isinstance(current_value, int):
            new_task["type"] = 0

        if isinstance(current_value, bool) or isinstance(current_value, str):
            new_task["type"] = 1

        # calculate the stepsize
        if new_task["in_time"] > 0:
            stepsize = (new_task["new_value"] - current_value) / (new_task["in_time"] / self._task_interval)
            new_task["stepsize"] = stepsize
            if stepsize != 0.0:
                self._tasks[id] = new_task
        else:
            new_task["type"] = 1
            new_task["stepsize"] = 0.0
            self._tasks[id] = new_task

        if new_task["type"] > 0:
            # calculate the stepsize
            new_task["stepsize"] = 0.0
            self._tasks[id] = new_task

    def remove_task(self, task_id: int) -> bool:
        if task_id in self._tasks.keys():
            del self._tasks[task_id]
            return True

        return False
    
    def remove_all_tasks(self):
        self._tasks = []

    def run_tasks(self):
        if self._task_interval_counter > self._task_interval:
            finished_tasks = []
            # reset the counter
            self._task_interval_counter = 0.0
            # run the tasks
            for id, task in self._tasks.items():
                # check if the task should be executed
                if task["at_time"] < self._task_interval and not task["running"]:
                    task["at_time"] = 0
                    # only do this for types which can not slowly change like booleans or strings
                    if task["type"] > 0:
                        task["current_value"] = task["new_value"]
                        self._set_value(task)
                        task["completed"] = True
                        finished_tasks.append(id)
                    else:
                        # get the task running
                        task["running"] = True
                else:
                    # decrease the time at
                    task["at_time"] -= self._task_interval

                # check whether the new value is already at the target value
                if task["type"] < 1 and task["running"]:
                    if abs(task["current_value"] - task["new_value"]) < abs(task["stepsize"]):
                        task["current_value"] = task["new_value"]
                        self._set_value(task)
                        task["stepsize"] = 0
                        task["completed"] = True
                        finished_tasks.append(id)
                    else:
                        task["current_value"] += task["stepsize"]
                        self._set_value(task)

            for ft in finished_tasks:
                del self._tasks[ft]

        if self.is_enabled:
            self._task_interval_counter += self._t

    def _set_value(self, task):
        if task["prop2"] is None:
            setattr(task["model"], task["prop1"], task["current_value"])
        else:
            p = getattr(task["model"], task["prop1"])
            p[task["prop2"]] = task["current_value"]

class Scaler():
    def __init__(self, model_ref: object) -> None:
        # -----------------------------------------------
        # initialize the independent properties
        self.reference_weight: float = 3.545            # reference weight for global scaling factor (kg)
        self.weight: float = 3.545                      # current weight (kg)
        self.height: float = 0.5                        # current height (m)

        # general scalers
        self.global_scale_factor: float = 1.0           # global scaling factor
        self.total_blood_volume_kg: float = 0.0         # total blood volume (L/kg)
        self.total_gas_volume_kg: float = 0.0           # total gas volume (L/kg)

        # Reference heartrate and pressures
        self.hr_ref: float = 125
        self.map_ref: float = 50

        # Scaling factors for breathing, metabolism, and cardiovascular system
        self.minute_volume_ref_scaling_factor: float = 1.0
        self.vt_rr_ratio_scaling_factor: float = 1.0
        self.vo2_scaling_factor: float = 1.0
        self.resp_q_scaling_factor: float = 1.0

        # Scaling factors of the heart and circulation
        self.el_min_atrial_factor: float = 1.0
        self.el_max_atrial_factor: float = 1.0
        self.el_min_ventricular_factor: float = 1.0
        self.el_max_ventricular_factor: float = 1.0
        self.el_min_cor_factor: float = 1.0
        self.el_max_cor_factor: float = 1.0
        self.el_base_pericardium_factor: float = 1.0
        self.el_base_syst_art_factor: float = 1.0
        self.el_base_syst_ven_factor: float = 1.0
        self.el_base_pulm_art_factor: float = 1.0
        self.el_base_pulm_ven_factor: float = 1.0
        self.el_base_cap_factor: float = 1.0
        self.el_base_da_factor: float = 1.0

        self.u_vol_atrial_factor: float = 1.0
        self.u_vol_ventricular_factor: float = 1.0
        self.u_vol_cor_factor: float = 1.0
        self.u_vol_pericardium_factor: float = 1.0
        self.u_vol_syst_art_factor: float = 1.0
        self.u_vol_syst_ven_factor: float = 1.0
        self.u_vol_pulm_art_factor: float = 1.0
        self.u_vol_pulm_ven_factor: float = 1.0
        self.u_vol_cap_factor: float = 1.0
        self.u_vol_da_factor: float = 1.0

        self.r_valve_factor: float = 1.0
        self.r_cor_factor: float = 1.0
        self.r_syst_art_factor: float = 1.0
        self.r_syst_ven_factor: float = 1.0
        self.r_pulm_art_factor: float = 1.0
        self.r_pulm_ven_factor: float = 1.0
        self.r_shunts_factor: float = 1.0
        self.r_da_factor: float = 1.0

        # Scaling factors of the lungs and chestwall
        self.el_base_lungs_factor: float = 1.0
        self.el_base_ds_factor: float = 1.0
        self.el_base_cw_factor: float = 1.0 
        self.el_base_thorax_factor: float = 1.0

        self.u_vol_lungs_factor: float = 1.0
        self.u_vol_ds_factor: float = 1.0
        self.u_vol_cw_factor: float = 1.0
        self.u_vol_thorax_factor: float = 1.0

        self.r_upper_airway_factor: float = 1.0
        self.r_lower_airway_factor: float = 1.0
        
        # -----------------------------------------------
        # initialize the dependent properties

        # -----------------------------------------------
        # initialize the local properties
        self._model_engine: object = model_ref          # object holding a reference to the model engine
        self._t: float = model_ref.modeling_stepsize    # modeling stepsize
        self._blood_containing_modeltypes: list = ["BloodCapacitance", "BloodTimeVaryingElastance"]

    def load_scaler_settings(self, scaler_settings: dict[str, any]) -> None:
        # set the properties of this model
        for key, value in scaler_settings.items():
            setattr(self, key, value)

        # scale the patient according to the settings
        self.scale_patient(self.weight, self.hr_ref, self.map_ref, self.total_blood_volume_kg, self.total_gas_volume_kg)
        
    def scale_patient(self, new_weight: float, hr_ref: float, map_ref: float, new_blood_volume_kg: float = 0.08, new_gas_volume_kg: float = 0.04) -> None:
        # calculate the global weight based scaling factor
        self.global_scale_factor = new_weight / self.reference_weight

        # store the new properties
        self._model_engine.weight = new_weight
        self.weight = new_weight
        self.hr_ref = hr_ref
        self.map_ref = map_ref

        # scale the blood volume
        self.scale_blood_volume(new_blood_volume_kg)

        # scale the gas volume
        self.scale_gas_volume(new_gas_volume_kg)

        # scale heart
        self.scale_heart()
        self.scale_heart_valves()
        self.scale_pericardium()
        self.scale_cor_resistors()

        # scale the arterial tree
        self.scale_syst_arteries()
        self.scale_syst_art_resistors()
        self.scale_pulm_arteries()
        self.scale_pulm_art_resistors()
        
        # scale the capillaries
        self.scale_capillaries()

        # scale the venous tree
        self.scale_syst_veins()
        self.scale_syst_ven_resistors()
        self.scale_pulm_veins()
        self.scale_pulm_ven_resistors()

        # scale the shunts
        self.scale_shunts()
        self.scale_ductus_arteriosus()
        self.scale_ductus_art_resistors()

        # scale the respiratory system
        self.scale_thorax()
        self.scale_chestwall()
        self.scale_lungs()
        self.scale_dead_space()
        self.scale_upper_airways()
        self.scale_lower_airways()

        # scale the other models
        self.scale_ans_hr(self.hr_ref)
        self.scale_ans_map(self.map_ref)
        self.scale_breathing()
        self.scale_metabolism()
        self.scale_mob()

        # print the scaling report
        print(f"Scaling model to {new_weight} kg => factor {self.global_scale_factor:.4f}")
        print(f"Scaling blood volume to {new_blood_volume_kg} L/kg = {self.total_blood_volume_kg * self._model_engine.weight:.4f} L")
        print(f"Scaling lung  volume to {new_gas_volume_kg} L/kg = {self.total_gas_volume_kg * self._model_engine.weight:.4f} L")

    def scale_blood_volume(self, new_blood_volume_kg: float):
        # get the current absolute total volume (L)
        current_blood_volume = self.get_total_blood_volume()

        # determine the new absolute blood volume (L)
        target_blood_volume = new_blood_volume_kg * self._model_engine.weight

        # calculate the change of the total volume
        scale_factor = target_blood_volume / current_blood_volume

        # change the volume of the blood containing models
        for _, m in self._model_engine.models.items():
            if (m.model_type in self._blood_containing_modeltypes):
                # scale the blood volume, the unstressed volume is scaled separately!
                m.vol = m.vol * scale_factor
                m.u_vol = m.u_vol * scale_factor

        # store the new total volume (L/kg)
        self.total_blood_volume_kg = self.get_total_blood_volume() / self._model_engine.weight

    def get_total_blood_volume(self) -> float:
        # declare an object for storing the volume
        _total_volume: float = 0.0
        # iterate over all blood containing models
        for _, m in self._model_engine.models.items():
            if (m.model_type in self._blood_containing_modeltypes):
                # if the model is enabled then at the current volume to the total volume
                if (m.is_enabled):
                    _total_volume += m.vol
                    
        # return the total blood volume
        return _total_volume

    def scale_gas_volume(self, new_gas_volume_kg: float):
        # get the current absolute total volume (L)
        current_gas_volume = self.get_total_gas_volume()

        # determine the new absolute blood volume (L)
        target_gas_volume = new_gas_volume_kg * self._model_engine.weight

        # calculate the change of the total volume
        scale_factor = target_gas_volume / current_gas_volume

        # change the volume of the blood containing models
        for _, m in self._model_engine.models.items():
            if (m.name in ["ALL", "ALR", "DS"]):
                # scale the gas volume, the unstressed volume is scaled separately!
                m.vol = m.vol * scale_factor
                m.u_vol = m.u_vol * scale_factor

        # store the new total volume (L/kg)
        self.total_gas_volume_kg = self.get_total_gas_volume() / self._model_engine.weight

    def get_total_gas_volume(self) -> float:
        # declare an object for storing the volume
        _total_volume: float = 0.0
        # iterate over all blood containing models
        for _, m in self._model_engine.models.items():
            if m.name in ["ALL", "ALR", "DS"]:
                # if the model is enabled then at the current volume to the total volume
                if (m.is_enabled):
                    _total_volume += m.vol
                    
        # return the total gas volume
        return _total_volume

    def scale_ans_hr(self, new_hr_ref: float):
        # store the new reference value
        self.hr_ref = new_hr_ref
        # adjust the ans reference values
        self.scale_ans(self.hr_ref, self.map_ref)
    
    def scale_ans_map(self, new_map_ref: float):
        # store the new reference value
        self.map_ref = new_map_ref
         # adjust the ans reference values
        self.scale_ans(self.hr_ref, self.map_ref)
    
    def scale_ans(self, hr_ref: float, map_ref: float):
        # store the new reference values
        self.hr_ref = hr_ref
        self.map_ref = map_ref
        # set the reference value on the Heart model
        self._model_engine.models["Heart"].heart_rate_ref = self.hr_ref
        # set the baroreceptor
        for m in self._model_engine.model_groups["baroreceptor"]:
            m.min_value = self.map_ref / 2.0
            m.set_value = self.map_ref
            m.max_value = self.map_ref * 2.0

    def scale_breathing(self):
        # as the reference minute volume and the vt_rr are already weight based this is an additional scaling
        # factor on top the weight based factor and we do not have to use the global scaling factor
        self._model_engine.models["Breathing"].minute_volume_ref_scaling_factor = self.minute_volume_ref_scaling_factor
        self._model_engine.models["Breathing"].vt_rr_ratio_scaling_factor = self.vt_rr_ratio_scaling_factor
    
    def scale_metabolism(self):
        # as the vo2 is already weight based this is an additional scaling factor 
        # and we do not have to use the global scaling factor
        self._model_engine.models["Metabolism"].vo2_scaling_factor = self.vo2_scaling_factor
        self._model_engine.models["Metabolism"].resp_q_scaling_factor = self.resp_q_scaling_factor
    
    def scale_mob(self):
        # the mob model is already completely weight based
        pass

    def scale_heart(self):
        # set the new factors on the heart chambers
        for m in self._model_engine.model_groups["heart_atria"]:
            m.el_min_scaling_factor = (1.0 / self.global_scale_factor) * self.el_min_atrial_factor
            m.el_max_scaling_factor = (1.0 / self.global_scale_factor) * self.el_max_atrial_factor
            m.u_vol_scaling_factor = self.u_vol_atrial_factor
        for m in self._model_engine.model_groups["heart_ventricles"]:
            m.el_min_scaling_factor = (1.0 / self.global_scale_factor) * self.el_min_ventricular_factor
            m.el_max_scaling_factor = (1.0 / self.global_scale_factor) * self.el_max_ventricular_factor
            m.u_vol_scaling_factor = self.u_vol_ventricular_factor
        for m in self._model_engine.model_groups["coronaries"]:
            m.el_min_scaling_factor = (1.0 / self.global_scale_factor) * self.el_min_cor_factor
            m.el_max_scaling_factor = (1.0 / self.global_scale_factor) * self.el_max_cor_factor
            m.u_vol_scaling_factor = self.u_vol_cor_factor

    def scale_heart_valves(self):
        # set the scale factors
        for m in self._model_engine.model_groups["heart_valves"]:
            m.r_scaling_factor = (1.0 / self.global_scale_factor) * self.r_valve_factor

    def scale_pericardium(self):
        # set the scaling factors
        for m in self._model_engine.model_groups["pericardium"]:
            m.el_base_scaling_factor = (1.0 / self.global_scale_factor) * self.el_base_pericardium_factor
            m.u_vol_scaling_factor = self.global_scale_factor * self.u_vol_pericardium_factor

    def scale_cor_resistors(self):
        # set the scale factors
        for m in self._model_engine.model_groups["cor_resistors"]:
            m.r_scaling_factor = (1.0 / self.global_scale_factor) * self.r_cor_factor

    def scale_syst_arteries(self):
        # set the scaling factors
        for m in self._model_engine.model_groups["syst_arteries"]:
            m.el_base_scaling_factor = (1.0 / self.global_scale_factor) * self.el_base_syst_art_factor
            m.u_vol_scaling_factor = self.u_vol_syst_art_factor

    def scale_syst_art_resistors(self):
        # set the scale factors
        for m in self._model_engine.model_groups["syst_art_resistors"]:
            m.r_scaling_factor = (1.0 / self.global_scale_factor) * self.r_syst_art_factor

    def scale_pulm_arteries(self):
        # set the scaling factors
        for m in self._model_engine.model_groups["pulm_arteries"]:
            m.el_base_scaling_factor = (1.0 / self.global_scale_factor) * self.el_base_pulm_art_factor
            m.u_vol_scaling_factor = self.u_vol_pulm_art_factor

    def scale_pulm_art_resistors(self):
        # set the scale factors
        for m in self._model_engine.model_groups["pulm_art_resistors"]:
            m.r_scaling_factor = (1.0 / self.global_scale_factor) * self.r_pulm_art_factor

    def scale_capillaries(self):
        # set the scaling factors
        for m in self._model_engine.model_groups["capillaries"]:
            m.el_base_scaling_factor = (1.0 / self.global_scale_factor) * self.el_base_cap_factor
            m.u_vol_scaling_factor = self.u_vol_cap_factor

    def scale_syst_veins(self):
        # set the scaling factors
        for m in self._model_engine.model_groups["syst_veins"]:
            m.el_base_scaling_factor = (1.0 / self.global_scale_factor) * self.el_base_syst_ven_factor
            m.u_vol_scaling_factor = self.u_vol_syst_ven_factor
    
    def scale_syst_ven_resistors(self):
        # set the scale factors
        for m in self._model_engine.model_groups["syst_ven_resistors"]:
            m.r_scaling_factor = (1.0 / self.global_scale_factor) * self.r_syst_ven_factor

    def scale_pulm_veins(self):
        # set the scaling factors
        for m in self._model_engine.model_groups["pulm_veins"]:
            m.el_base_scaling_factor = (1.0 / self.global_scale_factor) * self.el_base_pulm_ven_factor
            m.u_vol_scaling_factor = self.u_vol_pulm_ven_factor
    
    def scale_pulm_ven_resistors(self):
        # set the scale factors
        for m in self._model_engine.model_groups["pulm_ven_resistors"]:
            m.r_scaling_factor = (1.0 / self.global_scale_factor) * self.r_pulm_ven_factor

    def scale_shunts(self):
        # set the scale factors
        for m in self._model_engine.model_groups["shunts"]:
            m.r_scaling_factor = (1.0 / self.global_scale_factor) * self.r_shunts_factor

    def scale_ductus_arteriosus(self):
        # set the scaling factors
        for m in self._model_engine.model_groups["ductus_arteriosus"]:
            m.el_base_scaling_factor = (1.0 / self.global_scale_factor) * self.el_base_da_factor
            m.u_vol_scaling_factor = self.u_vol_da_factor

    def scale_ductus_art_resistors(self):
        # set the scale factors
        for m in self._model_engine.model_groups["ductus_art_resistors"]:
            m.r_scaling_factor = (1.0 / self.global_scale_factor) * self.r_da_factor

    def scale_thorax(self):
        # set the scaling factors
        for m in self._model_engine.model_groups["thorax"]:
            m.el_base_scaling_factor = (1.0 / self.global_scale_factor) * self.el_base_thorax_factor
            m.u_vol_scaling_factor = self.global_scale_factor * self.u_vol_thorax_factor

    def scale_chestwall(self):
        # set the scaling factors
        for m in self._model_engine.model_groups["chestwall"]:
            m.el_base_scaling_factor = (1.0 / self.global_scale_factor) * self.el_base_cw_factor
            m.u_vol_scaling_factor = self.global_scale_factor * self.u_vol_cw_factor

    def scale_lungs(self):
        # set the scaling factors
        for m in self._model_engine.model_groups["alveoli"]:
            m.el_base_scaling_factor = (1.0 / self.global_scale_factor) * self.el_base_lungs_factor
            m.u_vol_scaling_factor = self.u_vol_lungs_factor

    def scale_dead_space(self):
        # set the scaling factors
        for m in self._model_engine.model_groups["dead_space"]:
            m.el_base_scaling_factor = (1.0 / self.global_scale_factor) * self.el_base_ds_factor
            m.u_vol_scaling_factor = self.u_vol_ds_factor

    def scale_upper_airways(self):
        # set the scale factors
        for m in self._model_engine.model_groups["upper_airways"]:
            m.r_scaling_factor = (1.0 / self.global_scale_factor) * self.r_upper_airway_factor

    def scale_lower_airways(self):
        # set the scale factors
        for m in self._model_engine.model_groups["lower_airways"]:
            m.r_scaling_factor = (1.0 / self.global_scale_factor) * self.r_lower_airway_factor
    
class Plotter():
    def __init__(self) -> None:
        # dark mode flag
        self.dark_mode = False

        # plot line colors
        self.lines = ["r-", "b-", "g-", "c-", "m-", "y-", "k-", "w-"]
        self.plot_background_color = "#1E2029"
        self.plot_background_color = "#FFFFFF"
        self.plot_height = 3
        self.plot_dpi = 300
        self.plot_fontsize = 8
        self.plot_axis_color = "darkgray"

        # realtime variables
        self.plot_rt_background_color = "black"
        self.plot_rt_height = 4
        self.plot_rt_dpi = 300
        self.plot_rt_fontsize = 8
        self.plot_rt_axis_color = "darkgray"

        # define realtime intermediates
        self.x_rt = []
        self.y_rt = []
        self.ani = {}
        self.no_dp = 1200
        self.rt_time_window = 10.0
        self.rt_update_interval = 0.2
        self.combined = False
        self.rescale_counter = 0.0
        self.rescale_interval = 2.0
        self.rescale_enabled = False
        self.parameters_rt = []
        self.fig_rt = {}
        self.ax_rt = []
        self.axs_rt = []
        self.x = {}
        self.y = {}
        self.line_rt = []
        self.lines_rt = []
        self.xy = False
        self.x_prop = ""

    def draw_time_graph(self, collected_data, properties, sharey=False, combined=True, ylabel="", autoscale=True, ylowerlim=0, yupperlim=100, fill=True, fill_between=False, zeroline=False, fig_size_x=14, fig_size_y=3):
            parameters = properties
            no_parameters = 0
            # remove ncc ventricular from the list
            try:
                index = parameters.index("Heart.ncc_ventricular")
            except:
                index = -1
            
            if index > -1:
                parameters.pop(index)

            no_dp = len(collected_data)
            self.counter = no_dp
            x = np.zeros(no_dp)
            y = []

            for parameter in enumerate(parameters):
                y.append(np.zeros(no_dp))
                no_parameters += 1

            for index, t in enumerate(collected_data):
                x[index] = t["time"]

                for idx, parameter in enumerate(parameters):
                    y[idx][index] = t[parameter]

            # determine number of needed plots
            if self.dark_mode:
                plt.style.use("dark_background")

            # set the background color and erase the labels and headers

            if no_parameters == 1:
                combined = True

            if combined == False:
                fig, axs = plt.subplots(
                    nrows=no_parameters,
                    ncols=1,
                    figsize=(fig_size_x, fig_size_y * 0.75 * no_parameters),
                    sharex=True,
                    sharey=sharey,
                    constrained_layout=True,
                    dpi=self.plot_dpi / 3,
                )
                # Change to the desired color
                fig.patch.set_facecolor(self.plot_background_color)
                fig.set_label("")
                fig.canvas.header_visible = False

                # Change the fontsize as desired
                if no_parameters > 1:
                    for i, ax in enumerate(axs):
                        ax.tick_params(
                            axis="both", which="both", labelsize=self.plot_fontsize
                        )
                        ax.spines["right"].set_visible(False)
                        ax.spines["top"].set_visible(False)
                        ax.spines["bottom"].set_color(self.plot_axis_color)
                        ax.spines["left"].set_color(self.plot_axis_color)
                        ax.margins(x=0)
                        ax.plot(x, y[i], self.lines[i], linewidth=1)
                        ax.set_title(parameters[i], fontsize=self.plot_fontsize)
                        ax.set_xlabel("time (s)", fontsize=self.plot_fontsize)
                        ax.set_ylabel(ylabel, fontsize=self.plot_fontsize)
                        if not autoscale:
                            ax.set_ylim([ylowerlim, yupperlim])
                        if zeroline:
                            ax.hlines(0, np.amin(x), np.amax(x), linestyles="dashed")
                        if fill:
                            ax.fill_between(x, y[i], color="blue", alpha=0.3)

            if combined:
                fig = plt.figure(
                    figsize=(fig_size_x, fig_size_y),
                    dpi=self.plot_dpi / 3,
                    facecolor=self.plot_background_color,
                    tight_layout=True,
                )
                plt.tick_params(axis="both", which="both", labelsize=self.plot_fontsize)
                fig.patch.set_facecolor(self.plot_background_color)
                fig.set_label("")
                fig.canvas.header_visible = False

                ax = plt.gca()
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines["bottom"].set_color(self.plot_axis_color)
                ax.spines["left"].set_color(self.plot_axis_color)
                plt.margins(x=0, y=0)
                if not autoscale:
                    plt.ylim([ylowerlim, yupperlim])
                for index, parameter in enumerate(parameters):
                    # Subplot of figure 1 with id 211 the data (red line r-, first legend = parameter)
                    plt.plot(x, y[index], self.lines[index], linewidth=1, label=parameter)
                    if fill:
                        plt.fill_between(x, y[index], color="blue", alpha=0.3)
                if zeroline:
                    plt.hlines(0, np.amin(x), np.amax(x), linestyles="dashed")
                plt.xlabel("time (s)", fontsize=self.plot_fontsize)
                plt.ylabel(ylabel, fontsize=self.plot_fontsize)
                plt.xticks(fontsize=self.plot_fontsize)
                plt.yticks(fontsize=self.plot_fontsize)
                # Add a legend
                plt.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.22),
                    ncol=6,
                    fontsize=self.plot_fontsize,
                )
                if fill_between:
                    plt.fill_between(x, y[0], y[1], color="blue", alpha=0.1)

            plt.show()

    def draw_xy_graph(self, collected_data, property_x, property_y, fig_size_x=2, fig_size_y=2):
            no_dp = len(collected_data)
            x = np.zeros(no_dp)
            y = np.zeros(no_dp)

            for index, t in enumerate(collected_data):
                x[index] = t[property_x]
                y[index] = t[property_y]

            # determine number of needed plots
            if self.dark_mode:
                plt.style.use("dark_background")

            plt.figure(
                figsize=(fig_size_x, fig_size_y),
                dpi=self.plot_dpi / 1.5,
                facecolor=self.plot_background_color,
                tight_layout=True,
            )
            # Subplot of figure 1 with id 211 the data (red line r-, first legend = parameter)
            plt.plot(x, y, self.lines[0], linewidth=1)
            ax = plt.gca()
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_color(self.plot_axis_color)
            ax.spines["left"].set_color(self.plot_axis_color)

            plt.xlabel(property_x, fontsize=self.plot_fontsize / 3, labelpad=0)
            plt.ylabel(property_y, fontsize=self.plot_fontsize / 3, rotation=90, labelpad=1)
            plt.xticks(fontsize=self.plot_fontsize / 3)
            plt.yticks(fontsize=self.plot_fontsize / 3)
            plt.show()

class ModelEngine():
    def __init__(self, model_definition_json_filename: str = {}) -> None:
        # check whether the extension is already supplied or not
        if not ".json" in model_definition_json_filename:
            model_definition_json_filename = model_definition_json_filename + '.json'

        # load the model_definition json file and convert it to a python dictionary
        model_definition = {}
        with open(model_definition_json_filename, 'r') as json_file:
            model_definition = json.load(json_file)

        # define an object holding the model properties of the current model
        self.model_definition: dict = {}

        # define an object holding the entire model and submodels
        self.models: dict = {}

        # define an object holding the model groups
        self.model_groups: dict = {}

        # define an object holding the generated model data
        self.model_data: list = []

        # define an attribute holding the name of the model
        self.name: str = ""

        # define an attribute holding the description of the model
        self.description: str = ""

        # define an attribute holding the weight
        self.weight: float = 3.3

        # define an attribute holding the length in meters
        self.height: float = 0.50

        # define an attribute holding the body surface area
        self.bsa: float = 0.2

        # define an attribute holding the modeling stepsize
        self.modeling_stepsize: float = 0.0005

        # define an attribute holding the model time
        self.model_time_total: float = 0.0

        # define an object holding the  datacollector
        self._datacollector: object = None

        # define an object holding the task scheduler
        self._task_scheduler: object = None

        # define an object holding the data plotter
        self._plotter: object = None

        # define an object holding the model scaler
        self._scaler: object = None

        # define local attributes
        self.initialized: bool = False

        # if a model definition dictionary has been supplied then try to build the model
        if model_definition:
            self.build(model_definition)

    def build(self, model_definition: dict):
        # set the error counter = 0
        error_counter = 0

        # make sure the objects are empty
        self.models: dict = {}
        self.model_groups: dict = {}
        self.model_data: list = []
        self._datacollector: object = None
        self._task_scheduler: object = None
        self._scaler: object = None
        self.initialized = False

        # store the model definition in the model engine class
        self.model_definition = model_definition

        # try to process the model definiton dictionary
        try:
            # get the model attributes
            self.name = self.model_definition["name"]
            self.description = self.model_definition["description"]
            self.weight = self.model_definition["weight"]
            self.height = self.model_definition["height"]
            self.bsa = math.pow((self.weight * (self.height * 100.0) / 3600.0), 0.5)
            self.modeling_stepsize = self.model_definition["modeling_stepsize"]
            self.model_time_total = self.model_definition["model_time_total"]
            self.model_groups = self.model_definition["model_groups"].copy()
        except:
            # signal that the model definition dictionary failed
            print(f"The model definition dictionary could not be processed correctly!")
            self.initialized = False
            # terminate function
            return

        # instantiate all model components and put a reference to them in the models list
        for _, model in self.model_definition["models"].items():
            try:
                # get the desired sub model type from the dicitonary (e.g. BloodCapacitance or Heart model)
                model_type = model["model_type"]

                # try to instantiate a model class from the dicitonary and add the sub model to the models dictionary using the name of the submodel as key
                self.models[model["name"]] = globals()[model_type](self)

            except Exception as e:
                # print the error message and the sub model causing it and increase the error counter
                print(f"Model: {model['name']} failed to instantiate with error: {e}")
                error_counter += 1

        # instantiate a datacollector, taskscheduler and plotter object
        self._datacollector = Datacollector(self)
        self._task_scheduler = TaskScheduler(self)
        self._plotter = Plotter()
        self._scaler = Scaler(self)

        # replace contents of the model_groups with references to the models instead of the names
        model_groups_refs = {}
        # iterate over the model groups
        for group, model_list in self.model_groups.items():
            # declare a llist to hold the references to the actual models
            group_model_list = []
            # iterate over the group models list
            for model in model_list:
                # add a reference to the temporary group model list
                group_model_list.append(self.models[model])
            # make an entry for the group and set the group model list
            model_groups_refs[group] = group_model_list
        # copy the new model_groups dictionary to the old one
        self.model_groups = model_groups_refs.copy()
            
        # initialize all the submodels if there are no errors
        if error_counter == 0:
            init_errors = 0
            # initialize all components
            for model_name, model in self.models.items():
                # get all the properties of the sub model from the model definition dictionary
                model_props = self.model_definition["models"][model_name]
                try:
                    # try to initialize the sub model with the properties which we got from the model definition dictionary
                    model.init_model(**model_props)
                except Exception as e:
                    # print the error message and the sub model causing it and increase the error counter
                    print(f"Model: {model['name']} failed to initialize with error: {e}")
                    init_errors += 1

        # wrap up 
        if init_errors > 0 or error_counter > 0:
            self.initialized = False
        else:
            print(f" Model '{self.name}' initialized and ready for use. Have fun!")
            self.initialized = True

    def calculate(self, time_to_calculate: float = 10.0, performance: bool = True):
        # Calculate the number of steps of the model
        no_of_steps: int = int(time_to_calculate / self.modeling_stepsize)

        # Start the performance counter
        if performance:
            perf_start = perf_counter()

        # Do all model steps
        for _ in range(no_of_steps):
            # Execute the model step method of all models
            for model in self.models.values():
                if model.is_enabled:
                    model.step_model()

            # update the datacollector
            self._datacollector.collect_data(self.model_time_total)

            # run task
            self._task_scheduler.run_tasks()

            # Increase the model clock
            self.model_time_total += self.modeling_stepsize

        # Stop the performance counter
        if performance:
            # stop the performance counter
            perf_stop = perf_counter()

            # Store the performance metrics
            run_duration = perf_stop - perf_start
            step_duration = (run_duration / no_of_steps) * 1000

            # print the performance
            print(f"Model run in {run_duration:.4f} sec. Taking {no_of_steps} model steps with a step duration of {step_duration:.4f} sec.")
            print("")

    def set_properties(self, properties: list):
        # expected format of properties is a list of tuples with format [(property:str, new_value: number, in_time: number)] e.g. [("DA.el_base", 800)]

        # convert to list if a single tuple is given without list
        if isinstance(properties, tuple):
            properties = [properties]

        # process the properties change
        for property in properties:
            self.set_property(property=property[0], new_value=property[1], in_time=property[2])

    def set_property(self, property: str, new_value: float, in_time: float = 1.0,  at_time: float = 0.0) -> str:
        # define some placeholders
        task_id: int = random.randint(0, 1000)
        m: object = None
        p1: str = None
        p2: str = None

        # build a task scheduler object
        t = property.split(".")

        if len(t) < 4 and len(t) > 1:
            if len(t) == 2:
                m = t[0]
                p1 = t[1]
            if len(t) == 3:
                m = t[0]
                p1 = t[1]
                p2 = t[2]
        else:
            return False

        # define a task for the task scheduler
        task = {
            "id": task_id,
            "model": self.models[m],
            "prop1": p1,
            "prop2": p2,
            "new_value": new_value,
            "in_time": in_time,
            "at_time": at_time,
        }

        # pass the task to the scheduler
        self._task_scheduler.add_task(task)

        # return the task id
        return task_id

    def get_property(self, properties):
        if isinstance(properties, str):
            properties = [properties]

        # build a task scheduler object
        for property in properties:
            t = property.split(".")

            if len(t) < 4 and len(t) > 1:
                if len(t) == 2:
                    m = t[0]
                    p1 = t[1]
                    value = getattr(self.models[m], p1)
                    print(f"{property} = {value}")
                if len(t) == 3:
                    m = t[0]
                    p1 = t[1]
                    p2 = t[2]
                    value = getattr(self.models[m][p1], p2)
                    print(f"{property} = {value}")
            else:
                print("Invalid request!")
        
    def get_total_blood_volume(self, output=True):
        blood_containing_modeltypes = ["BloodCapacitance", "BloodTimeVaryingElastance"]
        total_volume: float = 0.0
        for _, m in self.models.items():
            if (m.model_type in blood_containing_modeltypes):
                if (m.is_enabled):
                    total_volume += m.vol
        
        if output:
            print(f"Total blood volume: {total_volume * 1000:.2f} ml = {total_volume / self.weight * 1000:.2f} ml/kg")

        return total_volume

    def get_bloodgas(self, comp=["AA"], output=True):
        result = {}
        if isinstance(comp, str):
            comp = [comp]

        for c in comp:
            self.models["Blood"].calc_blood_composition(self.models[c])
            result[c] = {
                "pH": self.models[c].ph,
                "pco2": self.models[c].pco2,
                "po2": self.models[c].po2,
                "hco3": self.models[c].hco3,
                "be": self.models[c].be,
                "so2": self.models[c].so2,
            }
            if output:
                print(f"Bloodgas in: {c}")
                print(f"pH   : {self.models[c].ph:.2f}")
                print(f"pco2 : {self.models[c].pco2:.1f} mmHg")
                print(f"po2  : {self.models[c].po2:.1f} mmHg")
                print(f"hco3 : {self.models[c].hco3:.1f} mmol/l")
                print(f"be   : {self.models[c].be:.1f} mmol/l")
                print(f"So2  : {self.models[c].so2:.1f}")
                print("")
            
        return result

    def analyze(self, properties, time_to_calculate=10, sampleinterval=0.005, calculate=True, weight_based=False):
        # define a result object
        result = {}

        # find the weight factor
        weight = 1.0
        if weight_based:
            weight = self.weight

        # add the ncc ventricular for correct analysis
        properties.insert(0, "Heart.ncc_ventricular")

        # make sure properties is a list
        if isinstance(properties, str):
            properties = [properties]

        # if calculation is necessary then do it
        if calculate:
            # first clear the watchllist and this also clears all data
            self._datacollector.clear_watchlist()

            # set the sample interval
            self._datacollector.set_sample_interval(sampleinterval)

            # add the properties to the watch_list
            for prop in properties:
                self._datacollector.add_to_watchlist(prop)

            # calculate the model steps
            self.calculate(time_to_calculate)

        no_dp = len(self._datacollector.collected_data)
        x = np.zeros(no_dp)
        y = []
        heartbeats = 0

        for parameter in enumerate(properties):
            y.append(np.zeros(no_dp))

        for index, t in enumerate(self._datacollector.collected_data):
            x[index] = t["time"]

            for idx, parameter in enumerate(properties):
                y[idx][index] = t[parameter]

        sv_message = False
        is_blood = False

        for idx, parameter in enumerate(properties):
            prop_category = parameter.split(sep=".")

            if (prop_category[1] == "pres" or prop_category[1] == "pres_in"):
                data = np.array(y[idx])
                max = round(np.amax(data), 5)
                min = round(np.amin(data), 5)
                mean = round((2 * min + max) / 3, 5)

                print("{:<16}: max {:10}, min {:10}, mean {:10} mmHg".format(parameter, max, min, mean))
                continue

            if prop_category[1] == "vol":
                data = np.array(y[idx])
                max = round(np.amax(data) * 1000 / weight, 5)
                min = round(np.amin(data) * 1000 / weight, 5)

                if weight_based:
                    print("{:<16}: max {:10}, min {:10} ml/kg".format(parameter, max, min))
                else:
                    print("{:<16}: max {:10}, min {:10} ml".format(parameter, max, min))
                continue

            if prop_category[1] == "ncc_ventricular":
                data = np.array(y[idx])
                heartbeats = np.count_nonzero(data == 1)
                continue

            if prop_category[1] == "flow":
                data = np.array(y[idx])
                data_forward = np.where(data > 0, data, 0)
                data_backward = np.where(data < 0, data, 0)

                t_start = x[0]
                t_end = x[-1]

                sum = np.sum(data) * 1000.0 / weight
                sum_forward = np.sum(data_forward) * 1000.0 / weight
                sum_backward = np.sum(data_backward) * 1000.0 / weight

                flow = (sum * sampleinterval / (t_end - t_start)) * 60.0
                flow = round(flow, 5)
                flow_forward = 0
                flow_backward = 0

                if flow != 0.0:
                    flow_forward = (sum_forward / sum) * flow
                    flow_forward = round(flow_forward, 5)

                    flow_backward = (sum_backward / sum) * flow
                    flow_backward = round(flow_backward, 5)

                is_blood = True
                if is_blood:
                    if sampleinterval == self.modeling_stepsize:
                        # use the no of heartbeats
                        bpm = (heartbeats / (t_end - t_start)) * 60
                    else:
                        if not sv_message:
                            print(f"Stroke volume calculation might be inaccurate. Try using a sampleinterval of {self.modeling_stepsize}")
                            sv_message = True
                        bpm = self.models["Heart"].heart_rate

                    sv = round(flow / bpm, 5)
                    if weight_based:
                        print("{:16}: net {:10}, forward {:10}, backward {:10} ml/kg/min, stroke volume: {:10} ml/kg, ".format(parameter, flow, flow_forward, flow_backward, sv))
                    else:
                        print("{:16}: net {:10}, forward {:10}, backward {:10} ml/min, stroke volume: {:10} ml, ".format(parameter, flow, flow_forward, flow_backward, sv))

                continue

            if prop_category[1] == "ncc_atrial":
                continue

            data = np.array(y[idx])
            max = round(np.amax(data), 5)
            min = round(np.amin(data), 5)

            print("{:<16}: max {:10} min {:10}".format(parameter, max, min))

    def plot(self,properties, time_to_calculate=10, combined=True, sharey=True, ylabel="", autoscale=True, ylowerlim=0, yupperlim=100, fill=True, fill_between=False, zeroline=False, sampleinterval=0.005, analyze=True, weight_based=False, fig_size_x=14, fig_size_y=2,):
        # first clear the watchllist and this also clears all data
        self._datacollector.clear_watchlist()

        # set the sample interval
        self._datacollector.set_sample_interval(sampleinterval)

        # add the property to the watchlist
        if isinstance(properties, str):
            properties = [properties]

        # add the properties to the watch_list
        for prop in properties:
            self._datacollector.add_to_watchlist(prop)

        # calculate the model steps
        if analyze:
            self.analyze(properties=properties, time_to_calculate=time_to_calculate, weight_based=weight_based, sampleinterval=sampleinterval)
        else:
            self.calculate(time_to_calculate)

        # get the collected data from the datacollector
        collected_data = self._datacollector.collected_data

        # plot the properties
        self._plotter.draw_time_graph(
            collected_data,
            properties,
            sharey,
            combined,
            ylabel,
            autoscale,
            ylowerlim,
            yupperlim,
            fill,
            fill_between,
            zeroline,
            fig_size_x,
            fig_size_y,
        )

    def plot_xy(self, property_x, property_y, time_to_calculate=2, sampleinterval=0.0005, fig_size_x=2, fig_size_y=2,):
        # first clear the watchllist and this also clears all data
        self._datacollector.clear_watchlist()

        # set the sample interval
        self._datacollector.set_sample_interval(sampleinterval)

        # add the properties to the watchlist
        self._datacollector.add_to_watchlist(property_x)
        self._datacollector.add_to_watchlist(property_y)

        # calculate the model steps
        self.calculate(time_to_calculate)

        # get the collected data from the datacollector
        collected_data = self._datacollector.collected_data

        self._plotter.draw_xy_graph(collected_data, property_x, property_y, fig_size_x, fig_size_y)

#----------------------------------------------------------------------------------------------------------------------------
# if running from an interactive python environment (e.g. jupyter/vs code) start with the following lines to import the model
'''
# import the explain model engine
from explain import ModelEngine

# initialize the model engine using the term_neonate model definition
model = ModelEngine("patients/term_neonate")

# plot the pressure inside the left ventricle and ascending aorta as an example
model.plot(["LV.pres_in", "AA.pres_in"], time_to_calculate=10)

# other model interaction code comes here
'''

# if not running inside a interactive python environment uncomment the next lines to initialize the model with the baseline_term_neonate dictionary
'''
# initialize the model engine using the term_neonate model definition
model = ModelEngine("patients/term_neonate")

# plot the pressure inside the left ventricle and ascending aorta as an example
model.plot(["LV.pres_in", "AA.pres_in"], time_to_calculate=10)

# other model interaction code comes here
'''

# run with: python3 explain.py (make sure that you have the matplotlib and numpy libraries installed). Again PyPy3 is MUCH faster :)




