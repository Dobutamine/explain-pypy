'''
Explain python code 
'''
import math
from abc import ABC, abstractmethod

# base model class
class BaseModelClass(ABC):
    """ 
        The base model class is the blueprint for all the model objects (classes). 
        It incorporates the properties and methods which all model objects must implement
        in order to be compatible with the model engine. 
    """

    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize independent properties which all models implement
        self.name: str = name                           # name of the model object
        self.description: str = ""                      # description in for documentation purposes
        self.is_enabled: bool = False                   # flag whether the model is enabled or not
        self.model_type: str = ""                       # holds the model type e.g. Capacitance

        # initialize local properties
        self._model_engine: object = model_ref          # object holding a reference to the model engine
        self._t: float = model_ref.modeling_stepsize    # setting the modeling stepsize
        self._is_initialized: bool = False              # flag whether the model is initialized or not

    def init_model(self, **args: dict[str, any]) -> None:
        # set the properties of this model as provided by the args dictionary
        for key, value in args.items():
            setattr(self, key, value)
        
        # flag that the model is initialized
        self._is_initialized = True

    def step_model(self) -> None:
        # this method is called by the model engine and if the model is enabled and initialized it will do the model calculations
        if self.is_enabled and self._is_initialized:
            self.calc_model()

    @abstractmethod
    def calc_model(self) -> None:
        # this method is abstract and must be implemented by subclasses
        pass

# explain core model components
class Capacitance(BaseModelClass):
    def __init__(self, model_ref: object, name: str) -> None:
        # initialize the BaseModelClass
        super().__init__(model_ref, name)

        # ---------------------------------------------------------------------------------------------------------------
        # initialize independent properties. The values of these properties are set by the ModelEngine.
        self.u_vol: float = 0.0                         # unstressed volume UV of the capacitance in (L)
        self.el_base: float = 0.0                       # baseline elastance E of the capacitance in (mmHg/L)
        self.el_k: float = 0.0                          # non-linear elastance factor K2 of the capacitance (unitless)
        self.pres_ext: float = 0.0                      # external pressure p2(t) (mmHg)

        # ---------------------------------------------------------------------------------------------------------------
        # initialize dependent properties
        self.vol: float = 0.0                           # volume v(t) (L)
        self.pres: float = 0.0                          # pressure p1(t) (mmHg)

    def calc_model(self) -> None:
        # calculate the pressure of the capacitance based on the current volume and the elastance properties
        self.pres = self.calc_pressure(self.vol, self.u_vol, self.el_base, self.el_k, self.pres_ext)

        # reset the external pressure as this is updated every model step
        self.pres_ext = 0.0

    def calc_pressure(self, v_t:float, UV:float, E: float, K2: float, p2_t) -> float:
        # calculate and return the pressure
        if (v_t - UV) >= 0:
            return K2 * (v_t - UV)**2 + E * (v_t - UV) +  p2_t
        else:   
            return -K2 * (v_t - UV)**2 + E * (v_t - UV) +  p2_t

    def volume_in(self, dv: float, comp_from: object) -> None:
        # change the volume of the capacitance by the amount dv. This function can be called by other model objects like a Resistor object
        self.vol += dv

    def volume_out(self, dv: float) -> None:
        # change the volume of the capacitance by the amount dv. This function can be called by other model objects like a Resistor object
        self.vol -= dv

class Resistor(BaseModelClass):
    '''
    A resistor is a model object which connects two Capacitances or TimeVaryingElastances and 
    models the flow between them based on the pressure difference and the resistance value.
    '''

    def __init__(self, model_ref: object, name: str) -> None:
        # initialize the BaseModelClass
        super().__init__(model_ref, name)

        # ---------------------------------------------------------------------------------------------------------------
        # initialize independent properties
        self.r_for:float  = 1.0                         # forward flow resistance Rf (mmHg/l*s)
        self.r_back: float = 1.0                        # backward flow resistance Rb (mmHg/l*s )
        self.r_k: float = 0.0                           # non linear resistance factor K1 (unitless)
        self.comp_from: str = ""                        # holds the name of the upstream component
        self.comp_to: str = ""                          # holds the name of the downstream component

        # -----------------------------------------------
        # initialize dependent properties
        self.flow: float = 0.0                          # flow f(t) (L/s)
        self._comp_from_ref: object = None              # holds a reference to the upstream component
        self._comp_to_ref: object = None                # holds a reference to the downstream component

    def init_model(self, **args: dict[str, any]) -> None:
        # call the parent init_model method to set the properties of this model as provided by the args dictionary
        super().init_model(**args)

        # get references to the connected components for faster access during model calculations
        self._comp_from_ref = self._model_engine.models[self.comp_from]
        self._comp_to_ref = self._model_engine.models[self.comp_to]

    def calc_model(self) -> None:
        # calculate the flow based on the pressures of the connected capacitances and the resistance values
        self.flow = self.calc_flow(
            self._comp_from_ref.pres, 
            self._comp_to_ref.pres, 
            self.r_for, 
            self.r_back, 
            self.r_k, 
            self.flow
        )

        # update the volumes of the connected capacitances based on the calculated flow
        self.update_volumes(self.flow)

    def calc_flow(self, p1_t: float, p2_t: float, Rf: float, Rb: float, K1: float, f_t: float) -> float:
        # calculate and return the flow based on the pressures and resistance values
        if (p1_t - p2_t) >= 0:
            return ((p1_t - p2_t) - K1 * (f_t ** 2)) / Rf
        else:
            return ((p1_t - p2_t) + K1 * (f_t ** 2)) / Rb
        
    def update_volumes(self, flow: float) -> None:
        if flow >= 0:
            self._comp_from_ref.volume_out(flow * self._t)
            self._comp_to_ref.volume_in(flow * self._t, self._comp_from_ref)
        else:
            self._comp_from_ref.volume_in(-flow * self._t, self._comp_to_ref)
            self._comp_to_ref.volume_out(-flow * self._t)

class Valve(BaseModelClass):
    '''
    A valve is a model object which connects two capacitances and models the flow between them based on the pressure difference and the resistance value.
    '''

    def __init__(self, model_ref: object, name: str) -> None:
        # initialize the BaseModelClass
        super().__init__(model_ref, name)

        # ---------------------------------------------------------------------------------------------------------------
        # initialize independent properties
        self.r_for:float  = 1.0                         # forward flow resistance Rf (mmHg/l*s)
        self.r_k: float = 0.0                           # non linear resistance factor K1 (unitless)
        self.comp_from: str = ""                        # holds the name of the upstream component
        self.comp_to: str = ""                          # holds the name of the downstream component

        # -----------------------------------------------
        # initialize dependent properties
        self.flow: float = 0.0                          # flow f(t) (L/s)

    def calc_model(self) -> None:
        # calculate the flow based on the pressures of the connected capacitances and the resistance values
        self.flow = self.calc_flow(
            self._model_engine.models[self.comp_from].pres, 
            self._model_engine.models[self.comp_to].pres, 
            self.r_for, 
            self.r_k, 
            self.flow
        )

        # update the volumes of the connected capacitances based on the calculated flow
        self.update_volumes(self.flow)

    def calc_flow(self, p1_t: float, p2_t: float, Rf: float, K1: float, f_t: float) -> float:
        # calculate and return the flow based on the pressures and resistance values
        if (p1_t - p2_t) >= 0:
            return ((p1_t - p2_t) - K1 * (f_t ** 2)) / Rf
        else:
            return 0.0
        
    def update_volumes(self, flow: float) -> None:
        if flow >= 0:
            self._comp_from_ref.volume_out(flow * self._t)
            self._comp_to_ref.volume_in(flow * self._t, self._comp_from_ref)
        else:
            self._comp_from_ref.volume_in(-flow * self._t, self._comp_to_ref)
            self._comp_to_ref.volume_out(-flow * self._t)

class Container(BaseModelClass):
    '''
    The Container model is a Capacitance model which can contain other volume containing models (e.g. Capacitance of TimeVaryingElastance).
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
        self.vol_extra: float = 0.0                     # additional volume of the container (L)
        self.contained_components: list = []            # list of names of models this Container contains

        # -----------------------------------------------
        # initialize dependent properties
        self.vol: float = 0.0                           # volume v(t) (L)
        self.pres: float = 0.0                          # pressure p1(t) (mmHg)

    def calc_model(self) -> None:
        # calculate the pressure of the Container based on the current volume and the elastance properties
        self.pres = self.calc_pressure(self.vol, self.u_vol, self.el_base, self.el_k, self.pres_ext)

        # update the external pressures of the contained components to match the Container's pressure
        for c in self.contained_components:
            self._model_engine.models[c].pres_ext += self.pres

        # reset the external pressure as this is updated every model step
        self.pres_ext = 0.0

    def calc_pressure(self, v_t:float, UV:float, E: float, K2: float, p2_t) -> float:
        # set the volume to the extra volume if there are no contained components
        v_t = self.vol_extra

        # calculate the total volume from contained components
        for c in self.contained_components:
            v_t += self._model_engine.models[c].vol

        # calculate and return the pressure
        if (v_t - UV) >= 0:
            return K2 * (v_t - UV)**2 + E * (v_t - UV) +  p2_t
        else:   
            return -K2 * (v_t - UV)**2 + E * (v_t - UV) +  p2_t

class TimeVaryingElastance(BaseModelClass):
    '''
    '''
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class
        super().__init__(model_ref, name)

        # ----------------------------------------------------------------
        # initialize independent properties which will be set when the init_model method is called
        self.u_vol: float = 0.0                         # unstressed volume UV of the capacitance in (L)
        self.el_min: float = 0.0                        # minimal elastance Emin in (mmHg/L)
        self.el_max: float = 0.0                        # maximal elastance emax(n) in (mmHg/L)
        self.el_k: float = 0.0                          # non-linear elastance factor K2 of the capacitance (unitless)
        self.pres_ext: float = 0.0                      # external pressure p2(t) in mmHg
        self.act_factor: float = 0.0                    # activation factor from the heart model (unitless)

        # -----------------------------------------------
        # initialize dependent properties
        self.vol: float = 0.0                           # volume v(t) (L)
        self.pres: float = 0.0                          # pressure p1(t) (mmHg)

    def calc_model(self) -> None:
        # calculate the pressure of the capacitance based on the current volume and the elastance properties
        self.pres = self.calc_pressure(self.vol, self.u_vol, self.el_base, self.el_k, self.pres_ext, self.act_factor)

        # reset the external pressure as this is updated every model step
        self.pres_ext = 0.0

    def calc_pressure(self, v_t:float, UV:float, E_min: float, e_max: float, K2: float, p2_t, a_t) -> float:
        # calculate the pressure
        p_ed_t = K2 * (v_t - UV)**2 + E_min * (v_t - UV)
        p_ms_t= e_max * (v_t - UV)
        return p_ms_t - p_ed_t * a_t + p_ed_t +  p2_t

    def volume_in(self, dv: float, comp_from: object) -> None:
        # change the volume of the capacitance by the amount dv. This function can be called by other model objects like a Resistor object
        self.vol += dv

    def volume_out(self, dv: float) -> None:
        # change the volume of the capacitance by the amount dv. This function can be called by other model objects like a Resistor object
        self.vol -= dv

class GasExchanger(BaseModelClass):
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class
        super().__init__(model_ref, name)

        # ---------------------------------------------------------------------------------------------------------------
        # initialize independent properties
        self.dif_o2: float = 0.0                        # diffusion constant for oxygen (mmol/mmHg * s)
        self.dif_co2: float = 0.0                       # diffusion constant for carbon dioxide (mmol/mmHg * s)
        self.comp_blood: str = ""                       # name of the blood component                
        self.comp_gas: str = ""                         # name of the gas component
        
        # ---------------------------------------------------------------------------------------------------------------
        # initialize dependent properties
        self.flux_o2 = 0.0                              # oxygen flux (mmol)
        self.flux_co2 = 0.0                             # carbon dioxide flux (mmol)

        # ---------------------------------------------------------------------------------------------------------------
        # local variables
        self._blood_comp_ref: object = None             # reference to the blood component
        self._gas_comp_ref: object = None               # reference to the gas component
        self._calc_blood_composition: object = None     # reference to the blood model blood composition calculation method


    def init_model(self, **args: dict[str, any]) -> None:
        # call the parent init_model method to set the properties of this model as provided by the args dictionary
        super().init_model(**args)

        # get references to the connected components for faster access during model calculations
        self._blood_comp_ref = self._model_engine.models[self.comp_blood]
        self._gas_comp_ref = self._model_engine.models[self.comp_gas]
        
        # store a reference to the calc_blood_composition function of the Blood model
        self._calc_blood_composition = self._model_engine.models["Blood"].calc_blood_composition

    def calc_model(self):
        # check whether the blood and gas components have volume
        if self._gas_comp_ref.vol == 0.0 or self._blood_comp_ref.vol == 0.0:
            return
        
        # set the blood composition of the blood component
        self._calc_blood_composition(self._blood_comp_ref)

         # get the partial pressures and gas concentrations from the components
        po2_blood = self._blood_comp_ref.po2
        pco2_blood = self._blood_comp_ref.pco2
        to2_blood = self._blood_comp_ref.to2
        tco2_blood = self._blood_comp_ref.tco2

        co2_gas = self._gas_comp_ref.co2
        cco2_gas = self._gas_comp_ref.cco2
        po2_gas = self._gas_comp_ref.po2
        pco2_gas = self._gas_comp_ref.pco2

        if self._blood_comp_ref.vol == 0.0:
            return
        
        # calculate the O2 flux from the blood to the gas compartment
        self.flux_o2 = ((po2_blood - po2_gas) * self.dif_o2 * self._t)

        # calculate the new O2 concentrations of the gas and blood compartments
        new_to2_blood = (to2_blood * self._blood_comp_ref.vol - self.flux_o2) / self._blood_comp_ref.vol
        if new_to2_blood < 0:
            new_to2_blood = 0.0

        new_co2_gas = (co2_gas * self._gas_comp_ref.vol + self.flux_o2) / self._gas_comp_ref.vol
        if new_co2_gas < 0:
            new_co2_gas = 0.0

        # calculate the CO2 flux from the blood to the gas compartment
        self.flux_co2 = ((pco2_blood - pco2_gas) * self.dif_co2 * self._t)

        # calculate the new CO2 concentrations of the gas and blood compartments
        new_tco2_blood = (tco2_blood * self._blood_comp_ref.vol - self.flux_co2) / self._blood_comp_ref.vol
        if new_tco2_blood < 0:
            new_tco2_blood = 0.0

        new_cco2_gas = (cco2_gas * self._gas_comp_ref.vol + self.flux_co2) / self._gas_comp_ref.vol
        if new_cco2_gas < 0:
            new_cco2_gas = 0.0

        # transfer the new concentrations
        self._blood_comp_ref.to2 = new_to2_blood
        self._blood_comp_ref.tco2 = new_tco2_blood
        self._gas_comp_ref.co2 = new_co2_gas
        self._gas_comp_ref.cco2 = new_cco2_gas

class Diffusor(BaseModelClass):
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # ---------------------------------------------------------------------------------------------------------------
        # initialize independent properties which will be set when the init_model method is called
        self.dif_o2: float = 0.01                       # diffusion constant for o2 (mmol/mmHg * s)
        self.dif_co2: float = 0.01                      # diffusion constant for co2 (mmol/mmHg * s)
        self.dif_solutes: dict = {}                     # diffusion constants for the different solutes (mmol/mmol * s)
        self.comp_blood1: str = ""                      # name of the first blood containing model
        self.comp_blood2: str = ""                      # name of the second blood containing model

        # ---------------------------------------------------------------------------------------------------------------
        # initialize dependent properties
        self.flux_o2 = 0.0                              # oxygen flux (mmol)
        self.flux_co2 = 0.0                             # carbon dioxide flux (mmol)

        # ---------------------------------------------------------------------------------------------------------------
        # local variables
        self._comp_blood1_ref: object = None            # holds a reference to the first blood containing model
        self._comp_blood2_ref: object = None            # holds a reference to the second blood containing model
        self._calc_blood_composition: object = None     # reference to the blood model blood composition calculation method

    def init_model(self, **args: dict[str, any]) -> None:
        # call the parent init_model method to set the properties of this model as provided by the args dictionary
        super().init_model(**args)

        # get references to the connected components for faster access during model calculations
        self._comp_blood1_ref = self._model_engine.models[self.comp_blood1]
        self._comp_blood2_ref = self._model_engine.models[self.comp_blood2]
        
        # store a reference to the calc_blood_composition function of the Blood model
        self._calc_blood_composition = self._model_engine.models["Blood"].calc_blood_composition

    def calc_model(self):
        # check whether the blood components have a volume
        if self._comp_blood1_ref.vol == 0.0 or self._comp_blood2_ref.vol == 0.0:
            return
        
        # calculate the blood composition of the blood components in this diffusor as we need the partial pressures for the gasses diffusion
        self._calc_blood_composition(self._comp_blood1_ref)
        self._calc_blood_composition(self._comp_blood2_ref)

        # diffuse the gasses where the diffusion is partial pressure driven
        do2 = (self._comp_blood1_ref.po2 - self._comp_blood2_ref.po2) * self.dif_o2 * self._t
        dco2 = (self._comp_blood1_ref.pco2 - self._comp_blood2_ref.pco2) * self.dif_co2 * self._t

        # update the concentrations
        self._comp_blood1_ref.to2 = ((self._comp_blood1_ref.to2 * self._comp_blood1_ref.vol) - do2) / self._comp_blood1_ref.vol
        self._comp_blood2_ref.to2 = ((self._comp_blood2_ref.to2 * self._comp_blood2_ref.vol) + do2) / self._comp_blood2_ref.vol

        self._comp_blood1_ref.tco2 = ((self._comp_blood1_ref.tco2 * self._comp_blood1_ref.vol) - dco2) / self._comp_blood1_ref.vol
        self._comp_blood2_ref.tco2 = ((self._comp_blood2_ref.tco2 * self._comp_blood2_ref.vol) + dco2) / self._comp_blood2_ref.vol

        # store the flux of o2 and co2
        self.flux_o2 = do2 / self._t
        self.flux_co2 = dco2 / self._t

        # diffuse the solutes where the diffusion is concentration gradient driven
        for sol, dif in self.dif_solutes.items():
            # diffuse the solute which is concentration driven
            dsol = (self._comp_blood1_ref.solutes[sol] - self._comp_blood2_ref.solutes[sol]) * dif * self._t
            # update the concentration
            self._comp_blood1_ref.solutes[sol] = ((self._comp_blood1_ref.solutes[sol] * self._comp_blood1_ref.vol) - dsol) / self._comp_blood1_ref.vol
            self._comp_blood2_ref.solutes[sol] = ((self._comp_blood2_ref.solutes[sol] * self._comp_blood2_ref.vol) + dsol) / self._comp_blood2_ref.vol

# explain from core models derived models
class BloodCapacitance(Capacitance):
    def __init__(self, model_ref: object, name: str) -> None:
        # initialize the parent class (Capacitance)
        super().__init__(model_ref, name)

        # ---------------------------------------------------------------------------------------------------------------
        # initialize additional independent properties. The values of these properties are set by the ModelEngine.
        self.temp: float = 0.0                          # blood temperature (dgs C)

        # ---------------------------------------------------------------------------------------------------------------
        # initialize additional dependent properties
        self.solutes: dict = {}                         # dictionary holding all solutes
        self.to2: float = 0.0                           # total oxygen concentration (mmol/l)
        self.tco2: float = 0.0                          # total carbon dioxide concentration (mmol/l)
        self.ph: float = -1.0                           # ph (unitless)
        self.pco2: float = -1.0                         # pco2 (mmHg)
        self.po2: float = -1.0                          # po2 (mmHg)
        self.so2: float = -1.0                          # o2 saturation
        self.hco3: float = -1.0                         # bicarbonate concentration (mmol/l)
        self.be: float = -1.0                           # base excess (mmol/l)

    # override the volume in method of the Capacitance class as this changes the concentrations of the gasses and solutes
    def volume_in(self, dv: float, comp_from: object) -> None:
        # change the volume of the capacitance by the amount dv. This function can be called by other model objects like a Resistor object
        self.vol += dv

        # process the gasses o2 and co2
        self.to2 += ((comp_from.to2 - self.to2) * dv) / self.vol
        self.tco2 += ((comp_from.tco2 - self.tco2) * dv) / self.vol

        # process the solutes
        for solute, conc in self.solutes.items():
            self.solutes[solute] += ((comp_from.solutes[solute] - conc) * dv) / self.vol

class BloodTimeVaryingElastance(TimeVaryingElastance):
    def __init__(self, model_ref: object, name: str) -> None:
        # initialize the parent class (Capacitance)
        super().__init__(model_ref, name)

        # ---------------------------------------------------------------------------------------------------------------
        # initialize independent properties. The values of these properties are set by the ModelEngine.
        self.temp: float = 0.0                          # blood temperature (dgs C)

        # ---------------------------------------------------------------------------------------------------------------
        # initialize dependent properties
        self.solutes: dict = {}                         # dictionary holding all solutes
        self.to2: float = 0.0                           # total oxygen concentration (mmol/l)
        self.tco2: float = 0.0                          # total carbon dioxide concentration (mmol/l)
        self.ph: float = -1.0                           # ph (unitless)
        self.pco2: float = -1.0                         # pco2 (mmHg)
        self.po2: float = -1.0                          # po2 (mmHg)
        self.so2: float = -1.0                          # o2 saturation
        self.hco3: float = -1.0                         # bicarbonate concentration (mmol/l)
        self.be: float = -1.0                           # base excess (mmol/l)

    # override the volume in method of the Capacitance class as this changes the concentrations of the gasses and solutes
    def volume_in(self, dv: float, comp_from: object) -> None:
        # change the volume of the capacitance by the amount dv. This function can be called by other model objects like a Resistor object
        self.vol += dv

        # process the gasses o2 and co2
        self.to2 += ((comp_from.to2 - self.to2) * dv) / self.vol
        self.tco2 += ((comp_from.tco2 - self.tco2) * dv) / self.vol

        # process the solutes
        for solute, conc in self.solutes.items():
            self.solutes[solute] += ((comp_from.solutes[solute] - conc) * dv) / self.vol

class GasCapacitance(Capacitance):
    def __init__(self, model_ref: object, name: str) -> None:
        # initialize the parent class (Capacitance)
        super().__init__(model_ref, name)

        # ---------------------------------------------------------------------------------------------------------------
        # initialize additional independent properties (these parameters are set to the correct value by the ModelEngine)
        self.pres_atm: float = 760                      # atmospheric pressure (mmHg)
        self.fixed_composition: float = False           # flag whether the gas composition of this capacitance can change

        # ---------------------------------------------------------------------------------------------------------------
        # initialize additional dependent properties
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
        
        # ---------------------------------------------------------------------------------------------------------------
        # local properties
        self._gas_constant: float = 62.36367            # ideal gas law gas constant (L·mmHg/(mol·K))
 
    def calc_model(self):
        # add heat to the gas
        self.add_heat()

        # add water vapour to the gas
        self.add_watervapour()

        # calculate the pressure of the capacitance based on the current volume and the elastance properties
        super().calc_model()

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

# explain core models
class Circulation(BaseModelClass):
    def __init__(self, model_ref: object, name: str = "") -> None:
        # initialize the base model class setting all the general properties of the model which all models have in common
        super().__init__(model_ref, name)

        # ---------------------------------------------------------------------------------------------------------------
        # initialize independent properties
        self.hr_ref: float = 110.0                      # reference heart rate (beats/minute)
        self.dhr_ans: float = 0.0                       # delta heart rate due to ANS input (beats/minute)
        self.dcont_ans: float = 0.0                     # delta contractility due to ANS input (unitless)
        self.dpvr_ans: float = 0.0                      # delta pulmonary vascular resistance due to ANS input (unitless)
        self.dsvr_ans: float = 0.0                      # delta systemic vascular resistance due to ANS input (unitless)
        self.duvol_ans: float = 0.0                     # delta unstressed volume due to ANS input (L)
        self.pq_time: float = 0.1                       # pq time (s)
        self.qrs_time: float = 0.075                    # qrs time (s)
        self.qt_time: float = 0.25                      # qt time (s)
        self.av_delay: float = 0.0005                   # delay in the AV-node (s)
        self.kn: float = 0.579                          # normalization constant of the ventricular activation function
                                
        # ---------------------------------------------------------------------------------------------------------------
        # initialize dependent properties
        self.hr: float = 120.0                          # calculated heart rate (beats/minute)
        self.qtc: float = 0.0                           # calculated qTc time (s)
        self.aaf: float = 0.0                           # atrial activation factor (unitless)
        self.vaf: float = 0.0                           # ventricular activation factor (unitless

        # ---------------------------------------------------------------------------------------------------------------
        # local state properties
        self.ncc_ventricular: int = 0                   # ventricular contraction counter   
        self.ncc_atrial: int = 0                        # atrial contraction counter
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

    def calc_model(self) -> None:
            # calculated the heart rate based on the reference heart rate and the ANS input
            self.hr = self.hr_ref + self.dhr_ans

            # prevent the heart rate from going below 10 bpm
            if self.hr < 10.0:
                self.hr = 10.0

            # calculate the ecg timings
            self.ecg_timers(self.hr)

            # calculate heart chamber activation factors
            self.heart_chamber_activations()

            # transfer the activation factors to the heart components
            self._model_engine.models["LA"].act_factor = self.aaf
            self._model_engine.models["RA"].act_factor = self.aaf
            self._model_engine.models["LV"].act_factor = self.vaf
            self._model_engine.models["RV"].act_factor = self.vaf
            self._model_engine.models["COR"].act_factor = self.vaf

    def ecg_timers(self, hr) -> None:
            # calculate the qTc
            self.qtc = self.calc_qtc(hr)

            # calculate the sinus node interval depending on the heart rate
            self._sa_node_interval = 60.0 / hr

            # has the sinus node period elapsed?
            if self._sa_node_timer > self._sa_node_interval:
                # reset the sinus node timer
                self._sa_node_timer = 0.0
                # signal that the pq-time starts running
                self._pq_running = True
                # reset the atrial activation curve counter
                self.ncc_atrial = -1

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
            if self._qt_timer > self.qtc:
                # reset the qt timer
                self._qt_timer = 0.0
                # signal that the qt timer has stopped
                self._qt_running = False
                # signal that the ventricles are coming out of their refractory state
                self._ventricle_is_refractory = False

            # increase the timers with the modeling stepsize as set by the model base class
            self._sa_node_timer += self._t

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

    def heart_chamber_activations(self) -> None:
        # calculate the atrial activation duration (half sine wave) normalized to 1
        _atrial_duration = self.pq_time / self._t
        if self.ncc_atrial >= 0 and self.ncc_atrial < _atrial_duration:
            self.aaf = math.sin(math.pi * (self.ncc_atrial / _atrial_duration))
        else:
            self.aaf = 0.0

        # calculate the ventricular activation factor (skewed half sine wave) normalized to 1
        _ventricular_duration = (self.qrs_time + self.qtc) / self._t
        if self.ncc_ventricular >= 0 and self.ncc_ventricular < _ventricular_duration:
            self.vaf = (self.ncc_ventricular / (self.kn * _ventricular_duration)) * math.sin(math.pi * (self.ncc_ventricular / _ventricular_duration))
        else:
            self.vaf = 0.0

    def calc_qtc(self, hr) -> float:
        if hr > 10.0:
            # Bazett's formula
            return self.qt_time * math.sqrt(60.0 / hr)
        else:
            return self.qt_time * 2.449

class Ventilation(BaseModelClass):
    pass

class Metabolism(BaseModelClass):
    pass

class Mob(BaseModelClass):
    pass

class Ans(BaseModelClass):
    pass

class Breathing(BaseModelClass):
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


