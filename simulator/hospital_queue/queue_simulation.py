#!/usr/bin/env python
# coding: utf-8

# # Simulacao leitos - camas limitadas

# In[1]:
# Import required modules

import simpy
import random
from random import expovariate, seed
import math
import pandas as pd
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from datetime import datetime

def run_queue_simulation(data,bar, bar_text, params={}):
    covid_data = data
    # In[3]:
    def novosLeitos(code, updated_beds):
        # consulta a tabela do CNES e a de notícias (google sheets) para ter uma simulação com inserção de leitos conforme passam os dias.
        # bem como consulta o número de leitos inicial

        beds_data = pd.read_csv('cnes_leitos.csv', sep=';')

        beds_data['total_beds_uci'] = beds_data['leito_sus_uti']

        normal_beds_sum = ['leito_sus_cirurgicos', 'leito_sus_clinicos',
                           'leito_sus_isolamento', 'leito_sus_outrasespec', 'leito_sus_uci']

        beds_data['total_beds'] = beds_data[normal_beds_sum].sum(axis=1)

        beds_data_filtered = beds_data[beds_data['IBGE'] == code]

        beds_simulation = beds_data_filtered[['total_beds', 'total_beds_uci']].sum(axis=0)

        beds_beginning = beds_simulation[0]

        beds_icu_beginning = beds_simulation[1]

        if not updated_beds:
            # caso o usuário escolha não atualiar, gera uma entrada nula de leitos (normais e de uti) futuros pra o dia 1
            matrix = np.array([[1], [0], [0]])

        else:
            # consulta a planilha e gera uma matriz numpy com as linhas contendo informações necessárias para a inserção de leitos
            # com timming, número de leitos normais e número de leitos UTI,  nesta ordem

            scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
                     "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

            creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)

            client = gspread.authorize(creds)

            sheet = client.open("Dados Projeto").worksheet('Simulador Filas Entrada')  # Open the spreadhseet

            data = sheet.get_all_records()  # Get a list of all records

            col0 = sheet.col_values(2)[5:]  # Datas
            col1 = sheet.col_values(3)[5:]  # Códigos
            col2 = sheet.col_values(6)[5:]  # Leitos Novos Normais
            col3 = sheet.col_values(11)[5:]  # Leitos Novos UTI

            dia0 = datetime(2020, 2, 29)

            for i in range(len(col0)):
                dia = datetime.strptime(col0[i], '%d/%m/%Y')
                col0[i] = (dia - dia0).days

            col0 = np.array(col0)
            col1 = np.array(col1)
            col2 = np.array(col2)
            col3 = np.array(col3)

            col1 = [int(numeric_string) for numeric_string in col1]
            col2 = [int(numeric_string) for numeric_string in col2]
            col3 = [int(numeric_string) for numeric_string in col3]

            col0 = np.reshape(col0, (1, np.size(col0)))
            col1 = np.reshape(col1, (1, np.size(col1)))
            col2 = np.reshape(col2, (1, np.size(col2)))
            col3 = np.reshape(col3, (1, np.size(col3)))

            col1 = np.array(col1)

            bool = (col1 == code)

            col0 = np.reshape(col0[bool], (1, np.size(col0[bool])))
            col2 = np.reshape(col2[bool], (1, np.size(col2[bool])))
            col3 = np.reshape(col3[bool], (1, np.size(col3[bool])))

            matrix = np.concatenate((col0, col2, col3), axis=0)

        return matrix, beds_beginning, beds_icu_beginning

    # In[19]:
    # # SIMULATION
    class g:
        """g holds Global variables. No individual instance is required"""

        has_covid = 1
        covid_cases = data
        covid_cases.head()
        cases_arriving = 1

        inter_arrival_time = 1 / cases_arriving  # Average time (hours) between arrivals

        los_covid = params["los_covid"]  # Average length of stay in hospital (hours)
        los_covid_uti = params["los_covid_icu"]

        sim_duration = covid_cases.shape[0]  # Duration of simulation (hours)
        audit_interval = 1  # Interval between audits (hours)

        # total_beds, total_beds_icu = load_beds()
        total_beds = params["total_beds"]
        total_beds_icu = params["total_beds_icu"]
        available_rate = params["available_rate"]
        icu_available_rate = params["available_rate_icu"]

        icu_rate = params["icu_rate"]
        icu_after_bed = params["icu_after_bed"]

        icu_death_rate = params["icu_death_rate"]  ### Adicionar nos Parâmetros
        icu_queue_death_rate = params["icu_queue_death_rate"]  ### Adicionar nos Parâmetros
        queue_death_rate = params["queue_death_rate"]

        beds = int(total_beds * available_rate)  # beds available
        # beds = int(total_beds)  # beds available
        icu_beds = int(total_beds_icu * icu_available_rate)  # icu beds available

    ##icu_beds = int(total_beds_icu) # icu beds available
    # In[20]:
    class Hospital:
        """
        Hospital class holds:
        1) Dictionary of patients present
        2) List of audit times
        3) List of beds occupied at each audit time
        4) Current total beds occupied
        5) Current total icu beds occupied
        6) Admissions to data
        Methods:
        __init__: Set up hospital instance
        audit: records number of beds occupied
        build_audit_report: builds audit report at end of run (calculate 5th, 50th
        and 95th percentile bed occupancy.
        chart: plot beds occupied over time (at end of run)
        """

        def __init__(self):
            """
            Constructor method for hospital class"
            Initialise object with attributes.
            """

            self.patients = {}  # Dictionary of patients present
            self.patients_in_queue = {}
            self.patients_in_icu_queue = {}

            self.patients_in_beds = {}
            self.patients_in_icu_beds = {}

            self.audit_time = []  # List of audit times
            self.audit_beds = []  # List of beds occupied at each audit time
            self.audit_icu_beds = []  # List of icu beds occupied at each audit time
            self.audit_queue = []
            self.audit_icu_queue = []

            self.audit_admissions = []
            self.audit_data = []

            self.audit_releases = []
            self.audit_queue_releases = []
            self.audit_icu_queue_releases = []
            self.audit_releases_ICU = []

            self.audit_bedToICU = []
            self.audit_ICUToBed = []  # TODO: review

            self.audit_ICUDeath = []  # TODO: review
            self.audit_queue_ICUDeath = []  # TODO: review
            self.audit_queue_Death = []  # TODO: review

            self.audit_admissions_normal_bed = []
            self.audit_admissions_icu = []

            self.audit_capacity = []
            self.audit_capacity_icu = []

            self.bed_count = 0  # Current total beds occupied
            self.bed_icu_count = 0
            self.queue_count = 0
            self.queue_icu_count = 0
            self.admissions = 0  # Admissions to data

            self.releases = 0
            self.queue_releases = 0
            self.icu_queue_releases = 0
            self.releases_ICU = 0

            self.bedToICU = 0
            self.ICUToBed = 0  # TODO: review

            self.ICUDeath = 0  # TODO: review
            self.queue_ICUDeath = 0
            self.queue_Death = 0

            self.admissions_normal_bed = 0
            self.admissions_icu = 0

            self.capacity = g.beds
            self.capacity_icu = g.icu_beds

            self.data = 0

            return

        def audit(self, time):
            """
            Audit method. When called appends current simulation time to audit_time
            list, and appends current bed count to audit_beds.
            """
            self.audit_time.append(time)
            self.audit_beds.append(self.bed_count)
            self.audit_icu_beds.append(self.bed_icu_count)
            self.audit_queue.append(self.queue_count)
            self.audit_icu_queue.append(self.queue_icu_count)

            self.audit_admissions.append(self.admissions)

            self.audit_releases.append(self.releases)
            self.audit_queue_releases.append(self.queue_releases)
            self.audit_icu_queue_releases.append(self.icu_queue_releases)
            self.audit_releases_ICU.append(self.releases_ICU)

            self.audit_bedToICU.append(self.bedToICU)
            self.audit_ICUToBed.append(self.ICUToBed)

            self.audit_queue_ICUDeath.append(self.queue_ICUDeath)
            self.audit_ICUDeath.append(self.ICUDeath)
            self.audit_queue_Death.append(self.queue_Death)

            self.audit_admissions_normal_bed.append(self.admissions_normal_bed)
            self.audit_admissions_icu.append(self.admissions_icu)

            self.audit_capacity.append(self.capacity)
            self.audit_capacity_icu.append(self.capacity_icu)

            self.audit_data.append(self.data)

            return

        def build_audit_report(self):
            """
            This method is called at end of run. It creates a pandas DataFrame,
            transfers audit times and bed counts to the DataFrame, and
            calculates/stores 5th, 50th and 95th percentiles.
            """
            self.audit_report = pd.DataFrame()

            self.audit_report['Time'] = self.audit_time

            self.audit_report['Data'] = self.audit_data

            self.audit_report['Occupied_beds'] = self.audit_beds

            self.audit_report['ICU_Occupied_beds'] = self.audit_icu_beds

            self.audit_report['Median_beds'] = self.audit_report['Occupied_beds'].quantile(0.5)

            self.audit_report['ICU_Median_beds'] = self.audit_report['ICU_Occupied_beds'].quantile(0.5)

            self.audit_report['Beds_5_percent'] = self.audit_report['Occupied_beds'].quantile(0.05)

            self.audit_report['ICU_Beds_5_percent'] = self.audit_report['ICU_Occupied_beds'].quantile(0.05)

            self.audit_report['Beds_95_percent'] = self.audit_report['Occupied_beds'].quantile(0.95)

            self.audit_report['ICU_Beds_95_percent'] = self.audit_report['ICU_Occupied_beds'].quantile(0.95)

            # queue
            self.audit_report['Queue'] = self.audit_queue

            self.audit_report['Median_queue'] = self.audit_report['Queue'].quantile(0.5)

            self.audit_report['Queue_5_percent'] = self.audit_report['Queue'].quantile(0.05)

            self.audit_report['Queue_95_percent'] = self.audit_report['Queue'].quantile(0.95)

            # icu queue
            self.audit_report['ICU_Queue'] = self.audit_icu_queue

            self.audit_report['ICU_Median_queue'] = self.audit_report['ICU_Queue'].quantile(0.5)

            self.audit_report['ICU_Queue_5_percent'] = self.audit_report['ICU_Queue'].quantile(0.05)

            self.audit_report['ICU_Queue_95_percent'] = self.audit_report['ICU_Queue'].quantile(0.95)

            self.audit_report['Admissons'] = self.audit_admissions

            self.audit_report['AdmissonsNormalBed'] = self.audit_admissions_normal_bed

            self.audit_report['AdmissonsICU'] = self.audit_admissions_icu

            self.audit_report['Releases'] = self.audit_releases

            self.audit_report['ICU_Releases'] = self.audit_releases_ICU

            self.audit_report['Queue Releases'] = self.audit_queue_releases

            self.audit_report['ICU Queue Releases'] = self.audit_icu_queue_releases

            self.audit_report['bedToICU'] = self.audit_bedToICU

            self.audit_report['ICUToBed'] = self.audit_ICUToBed

            self.audit_report['ICUDeath'] = self.audit_ICUDeath

            self.audit_report['Queue ICUDeath'] = self.audit_queue_ICUDeath

            self.audit_report['Queue Death'] = self.audit_queue_Death

            self.audit_report['Capacity'] = self.audit_capacity

            self.audit_report['Capacity_ICU'] = self.audit_capacity_icu

            return self.audit_report

        def chart(self):
            """
            This method is called at end of run. It plots beds occupancy over the
            model run, with 5%, 50% and 95% percentiles.
            """

            # Plot occupied beds

            plt.plot(self.audit_report['Time'],
                     self.audit_report['Occupied_beds'],
                     color='k',
                     marker='o',
                     linestyle='solid',
                     markevery=1,
                     label='Occupied beds')

            plt.plot(self.audit_report['Time'],
                     self.audit_report['Beds_5_percent'],
                     color='0.5',
                     linestyle='dashdot',
                     markevery=1,
                     label='5th percentile')

            plt.plot(self.audit_report['Time'],
                     self.audit_report['Median_beds'],
                     color='0.5',
                     linestyle='dashed',
                     label='Median')

            plt.plot(self.audit_report['Time'],
                     self.audit_report['Beds_95_percent'],
                     color='0.5',
                     linestyle='dashdot',
                     label='95th percentile')

            plt.xlabel('Day')
            plt.ylabel('Occupied beds')
            plt.title(
                'Occupied beds (individual days with 5th, 50th and 95th ' +
                'percentiles)')
            plt.legend()
            plt.show()

            # Plot occupied icu beds

            plt.plot(self.audit_report['Time'],
                     self.audit_report['ICU_Occupied_beds'],
                     color='k',
                     marker='o',
                     linestyle='solid',
                     markevery=1,
                     label='Occupied icu beds')

            plt.plot(self.audit_report['Time'],
                     self.audit_report['ICU_Beds_5_percent'],
                     color='0.5',
                     linestyle='dashdot',
                     markevery=1,
                     label='5th percentile')

            plt.plot(self.audit_report['Time'],
                     self.audit_report['ICU_Median_beds'],
                     color='0.5',
                     linestyle='dashed',
                     label='Median')

            plt.plot(self.audit_report['Time'],
                     self.audit_report['ICU_Beds_95_percent'],
                     color='0.5',
                     linestyle='dashdot',
                     label='95th percentile')

            plt.xlabel('Day')
            plt.ylabel('Occupied icu beds')
            plt.title(
                'Occupied icu beds (individual days with 5th, 50th and 95th ' +
                'percentiles)')
            plt.legend()
            plt.show()

            # Plot queue for beds

            plt.plot(self.audit_report['Time'],
                     self.audit_report['Queue'],
                     color='k',
                     marker='o',
                     linestyle='solid',
                     markevery=1, label='Occupied beds')

            plt.plot(self.audit_report['Time'],
                     self.audit_report['Queue_5_percent'],
                     color='0.5',
                     linestyle='dashdot',
                     markevery=1,
                     label='5th percentile')

            plt.plot(self.audit_report['Time'],
                     self.audit_report['Median_queue'],
                     color='0.5',
                     linestyle='dashed',
                     label='Median')

            plt.plot(self.audit_report['Time'],
                     self.audit_report['Queue_95_percent'],
                     color='0.5',
                     linestyle='dashdot',
                     label='95th percentile')

            plt.xlabel('Day')
            plt.ylabel('Queue for beds')
            plt.title('Queue for beds (individual days with 5th, 50th and 95th' +
                      ' percentiles)')
            plt.legend()
            plt.show()

            # Plot queue for ICU beds

            plt.plot(self.audit_report['Time'],
                     self.audit_report['ICU_Queue'],
                     color='k',
                     marker='o',
                     linestyle='solid',
                     markevery=1, label='Occupied ICU beds')

            plt.plot(self.audit_report['Time'],
                     self.audit_report['ICU_Queue_5_percent'],
                     color='0.5',
                     linestyle='dashdot',
                     markevery=1,
                     label='5th percentile')

            plt.plot(self.audit_report['Time'],
                     self.audit_report['ICU_Median_queue'],
                     color='0.5',
                     linestyle='dashed',
                     label='Median')

            plt.plot(self.audit_report['Time'],
                     self.audit_report['ICU_Queue_95_percent'],
                     color='0.5',
                     linestyle='dashdot',
                     label='95th percentile')

            plt.xlabel('Day')
            plt.ylabel('Queue for ICU bed')
            plt.title('Queue for ICU bed (individual days with 5th, 50th and 95th' +
                      ' percentiles)')
            plt.legend()
            plt.show()

            return

    class Model:
        """
        The main model class.
        The model class contains the model environment. The modelling environment
        is set up, and patient arrival and audit processes initiated. Patient
        arrival triggers a spell for that patient in hospital. Arrivals and audit
        continue for the duration of the model run. The audit is then
        summarised and bed occupancy (with 5th, 50th and 95th percentiles) plotted.
        Methods are:
        __init__: Set up model instance
        audit_beds: call for bed audit at regular intervals (after initial delay
        for model warm-up)
        new_admission: trigger new admissions to hospital at regular intervals.
        Call for patient generation with patient id and length of stay, then call
        for patient spell in hospital.
        run: Controls the main model run. Initialises model and patient arrival and
        audit processes. Instigates the run. At end of run calls for an audit
        summary and bed occupancy plot.
        spell_gen: stores patient in hospital patient list and bed queue
        dictionaries, waits for bed resource to become available, then removes
        patient from bed queue dictionary and adds patient to hospital bed
        dictionary and increments beds occupied. Waits for the patient length of
        stay in the hospital and then decrements beds occupied and removes patient
        from hospital patient dictionary and beds occupied dictionary.
        """

        def __init__(self):
            """
            Constructor class for new model.
            """
            self.env = simpy.Environment()

            return

        def audit_beds(self, delay):
            """
            Bed audit process. Begins by applying delay, then calls for audit at
            intervals set in g.audit_interval
            :param delay: delay (days) at start of model run for model warm-up.
            """

            # Delay first audit
            yield self.env.timeout(delay)

            # Continually generate audit requests until end of model run
            while True:
                # Call audit (pass simulation time to hospital.audit)
                self.hospital.data = g.covid_cases['day'][math.floor(self.env.now)]
                self.hospital.audit(self.env.now)
                # Delay until next call
                yield self.env.timeout(g.audit_interval)

            return

        def halt(self, delay):
            """
            Break condition for the simulation.
            :param delay: number of days after complete resource saturation in
                          which to stop simulation
            """

            while True:

                if  self.env.now > g.sim_duration - 2:
                    break

                # Bed and ICU saturation condition
                if self.hospital.queue_icu_count > 1 and \
                        self.hospital.queue_count > 1:

                    print('Saturation reached at %f' % self.env.now)
                    # Delay before coming to an end
                    yield self.env.timeout(delay)
                    # print('Simulation stopped at %f' % self.env.now)

                    break
                else:
                    yield self.env.timeout(1)

            return

        def occupy_future_beds(self, time):

            with self.resources.beds.request() as req:
                yield self.env.timeout(time)

            self.hospital.capacity += 1

            return

        def occupy_future_beds_icu(self, time):

            with self.resources_icu.icu_beds.request() as icu_req:
                yield self.env.timeout(time)

            self.hospital.capacity_icu += 1

            return

        def new_admission(self, interarrival_time):
            """
            New admissions to hospital.
            :param interarrival_time: average time (days) between arrivals
            :param los: average length of stay (days)
            """
            # TODO: REMOVE LOS AND LOS_UTI
            while True:
                # Increment hospital admissions count
                self.hospital.admissions += 1

                # Generate new patient object (from Patient class). Give patient id
                # and set length of stay from inverse exponential distribution).
                p = Patient(patient_id=self.hospital.admissions,
                            los=random.expovariate(1 / g.los_covid),
                            los_uti=random.expovariate(1 / g.los_covid_uti))
                # else:
                #     p = Patient(patient_id=self.hospital.admissions,
                #                 los=random.expovariate(1 / los),
                #                 los_uti=random.expovariate(1 / los_uti))

                # print('Patient %d arriving %7.2f, admissions count %d' %(p.id,self.env.now,self.hospital.admissions))

                # Add patient to hospital patient dictionary
                self.hospital.patients[p.id] = p

                # Generate a patient spell in hospital (by calling spell method).
                # This triggers a patient admission and allows the next arrival to
                # be set before the paitent spell is finished
                self.spell = self.spell_gen(p)
                self.env.process(self.spell)

                # Set and call delay before looping back to new patient admission

                inter_arrival_covid = 1 / g.covid_cases['hospitalizados'][math.floor(self.env.now)]
                next_admission = random.expovariate(1 / interarrival_time + 1 / inter_arrival_covid)
                ##next_admission = random.expovariate(1 / ((interarrival_time + inter_arrival_covid)/2) )

                bar.progress(round(float(self.env.now / g.sim_duration), 2))
                bar_text.text(f"Processando dia {math.floor(self.env.now)}")

                yield self.env.timeout(next_admission)

            return

        def spell_gen(self, p):
            """
            Patient hospital stay generator. Increment bed count, wait for patient
            length of stay to complete, then decrement bed count and remove patient
            from hospital patient dictionary
            :param p: patient object (contains length of stay for patient)
            """
            # The following 'with' defines the required resources and automatically
            # releases resources when no longer required

            # screening - bed or icu bed
            is_icu = 1 if random.uniform(0, 1) > (1 - g.icu_rate) else 0

            # bed
            if is_icu == 0:

                self.hospital.admissions_normal_bed += 1

                # Increment queue count
                self.hospital.queue_count += 1
                # print('Patient %d arriving queue %7.2f, queue count %d' %(p.id,self.env.now,self.hospital.queue_count))
                # print('Occupied Beds %d'%(self.hospital.bed_count))

                # Add patient to dictionary of queuing patients. This is not used
                # further in this model.
                self.hospital.patients_in_queue[p.id] = p

                # Yield resource request. Sim continues after yield when resources
                # are vailable (so there is no delay if resources are immediately
                # available)
                with self.resources.beds.request(priority=2) as req:

                    final = yield req | self.env.timeout(p.los)

                    if not (req in final):

                        is_dead = 1 if random.uniform(0, 1) < g.queue_death_rate else 0

                        if is_dead == 1:
                            self.hospital.queue_count -= 1
                            # print('Patient %d leaving icu bed %7.2f, icu bed count %d' %(p.id,self.env.now,self.hospital.bed_icu_count))
                            del self.hospital.patients_in_queue[p.id]
                            del self.hospital.patients[p.id]
                            self.resources.beds.release(req)
                            self.hospital.queue_Death += 1

                        else:
                            self.hospital.queue_count -= 1
                            # print('Patient %d leaving icu bed %7.2f, icu bed count %d' %(p.id,self.env.now,self.hospital.bed_icu_count))
                            del self.hospital.patients_in_queue[p.id]
                            del self.hospital.patients[p.id]
                            self.resources.beds.release(req)
                            self.hospital.queue_releases += 1

                    else:
                        # Resource now available. Remove from queue count and dictionary of
                        # queued objects
                        self.hospital.queue_count -= 1
                        del self.hospital.patients_in_queue[p.id]
                        # print('Patient %d leaving queue %7.2f, queue count %d' %(p.id,self.env.now,self.hospital.queue_count))

                        # Add to count of patients in beds and to dictionary of patients in
                        # beds
                        self.hospital.patients_in_beds[p.id] = p
                        self.hospital.bed_count += 1
                        # print('Patient %d arriving bed %7.2f, bed count %d' %(p.id,self.env.now,self.hospital.bed_count))

                        # Trigger length of stay delay
                        yield self.env.timeout(p.los)

                        # needs icu
                        after_is_icu = 1 if random.uniform(0, 1) > (1 - g.icu_after_bed) else 0

                        if after_is_icu == 1:

                            # Increment queue count
                            self.hospital.queue_icu_count += 1
                            # print('Patient %d waiting in icu queue %7.2f, queue icu count %d' %(p.id,self.env.now,self.hospital.queue_icu_count))

                            # Add patient to dictionary of icu queuing patients. This is not used
                            # further in this model.
                            self.hospital.patients_in_icu_queue[p.id] = p

                            # Yield resource request. Sim continues after yield when resources
                            # are vailable (so there is no delay if resources are immediately
                            # available)

                            with self.resources_icu.icu_beds.request(priority=1) as icu_req:

                                final = yield icu_req | self.env.timeout(p.los_uti)

                                if not (icu_req in final):

                                    is_dead = 1 if random.uniform(0, 1) < g.icu_queue_death_rate else 0

                                    if is_dead == 1:
                                        self.hospital.queue_icu_count -= 1
                                        self.resources_icu.icu_beds.release(icu_req)
                                        del self.hospital.patients_in_icu_queue[p.id]
                                        # print('Patient %d leaving icu bed %7.2f, icu bed count %d' %(p.id,self.env.now,self.hospital.bed_icu_count))

                                        self.hospital.bed_count -= 1
                                        del self.hospital.patients_in_beds[p.id]

                                        del self.hospital.patients[p.id]
                                        self.hospital.queue_ICUDeath += 1

                                    else:
                                        self.hospital.queue_icu_count -= 1
                                        self.resources_icu.icu_beds.release(icu_req)
                                        del self.hospital.patients_in_icu_queue[p.id]
                                        # print('Patient %d leaving icu bed %7.2f, icu bed count %d' %(p.id,self.env.now,self.hospital.bed_icu_count))

                                        self.hospital.bed_count -= 1
                                        del self.hospital.patients_in_beds[p.id]

                                        del self.hospital.patients[p.id]
                                        self.hospital.icu_queue_releases += 1
                                else:

                                    # Resource now available. Remove from queue count and dictionary of
                                    # queued objects
                                    self.hospital.bed_count -= 1
                                    del self.hospital.patients_in_beds[p.id]
                                    # TODO: review bed release
                                    self.resources.beds.release(req)
                                    # print('Patient %d leaving bed %7.2f, queue bed %d' %(p.id,self.env.now,self.hospital.bed_count))

                                    # Increment queue count
                                    self.hospital.queue_icu_count -= 1
                                    del self.hospital.patients_in_icu_queue[p.id]
                                    # print('Patient %d leaving icu queue %7.2f, queue icu count %d' %(p.id,self.env.now,self.hospital.queue_icu_count))

                                    # Add to count of patients in icu beds and to dictionary of patients in
                                    # icu beds
                                    self.hospital.patients_in_icu_beds[p.id] = p
                                    self.hospital.bed_icu_count += 1
                                    # print('Patient %d arriving icu bed %7.2f, icu bed count %d' %(p.id,self.env.now,self.hospital.bed_icu_count))

                                    self.hospital.bedToICU += 1

                                    # Trigger length of stay delay
                                    yield self.env.timeout(p.los_uti)

                                    is_dead = 1 if random.uniform(0, 1) < g.icu_death_rate else 0

                                    if is_dead == 1:

                                        self.hospital.bed_icu_count -= 1
                                        # print('Patient %d leaving icu bed %7.2f, icu bed count %d' %(p.id,self.env.now,self.hospital.bed_icu_count))
                                        del self.hospital.patients_in_icu_beds[p.id]
                                        del self.hospital.patients[p.id]
                                        self.resources_icu.icu_beds.release(icu_req)
                                        self.hospital.ICUDeath += 1

                                    else:

                                        # TODO: Review pacients returning to regular beds
                                        # <---
                                        # Returns to bed

                                        # Increment queue count
                                        self.hospital.queue_count += 1
                                        # print('Patient %d arriving queue %7.2f, queue count %d' %(p.id,self.env.now,self.hospital.queue_count))
                                        # print('Occupied Beds %d'%(self.hospital.bed_count))

                                        # Add patient to dictionary of queuing patients. This is not used
                                        # further in this model.
                                        self.hospital.patients_in_queue[p.id] = p

                                        # Yield resource request. Sim continues after yield when resources
                                        # are vailable (so there is no delay if resources are immediately
                                        # available)
                                        with self.resources.beds.request(priority=1) as req:

                                            final = yield req | self.env.timeout(p.los)

                                            if not (req in final):

                                                is_dead = 1 if random.uniform(0, 1) < g.queue_death_rate else 0

                                                if is_dead == 1:
                                                    self.hospital.queue_count -= 1
                                                    # print('Patient %d leaving icu bed %7.2f, icu bed count %d' %(p.id,self.env.now,self.hospital.bed_icu_count))
                                                    del self.hospital.patients_in_queue[p.id]
                                                    self.resources.beds.release(req)

                                                    self.hospital.bed_icu_count -= 1
                                                    del self.hospital.patients_in_icu_beds[p.id]

                                                    del self.hospital.patients[p.id]
                                                    self.hospital.queue_Death += 1

                                                else:
                                                    self.hospital.queue_count -= 1
                                                    # print('Patient %d leaving icu bed %7.2f, icu bed count %d' %(p.id,self.env.now,self.hospital.bed_icu_count))
                                                    del self.hospital.patients_in_queue[p.id]
                                                    self.resources.beds.release(req)

                                                    self.hospital.bed_icu_count -= 1
                                                    del self.hospital.patients_in_icu_beds[p.id]

                                                    del self.hospital.patients[p.id]
                                                    self.hospital.queue_releases += 1

                                            else:

                                                # Resource now available. Remove from queue count and dictionary of
                                                # queued objects
                                                self.hospital.queue_count -= 1
                                                del self.hospital.patients_in_queue[p.id]
                                                # print('Patient %d leaving queue %7.2f, queue count %d' %(p.id,self.env.now,self.hospital.queue_count))
                                                # --->
                                                self.hospital.bed_icu_count -= 1
                                                # print('Patient %d leaving icu bed %7.2f, icu bed count %d' %(p.id,self.env.now,self.hospital.bed_icu_count))
                                                del self.hospital.patients_in_icu_beds[p.id]
                                                self.resources_icu.icu_beds.release(icu_req)

                                                # TODO: Review pacient conut
                                                # <---
                                                # Add to count of patients in beds and to dictionary of patients in
                                                # beds
                                                self.hospital.patients_in_beds[p.id] = p
                                                self.hospital.bed_count += 1
                                                # print('Patient %d arriving bed %7.2f, bed count %d' %(p.id,self.env.now,self.hospital.bed_count))

                                                self.hospital.ICUToBed += 1

                                                # Trigger length of stay delay
                                                yield self.env.timeout(p.los)

                                                # Length of stay complete. Remove patient from counts and
                                                # dictionaries
                                                self.hospital.bed_count -= 1
                                                # print('Patient %d leaving bed %7.2f, bed count %d' %(p.id,self.env.now,self.hospital.bed_count))
                                                # print('Patient %d released %7.2f, bed count %d' %(p.id,self.env.now,self.hospital.bed_count))
                                                del self.hospital.patients_in_beds[p.id]
                                                self.resources.beds.release(req)
                                                del self.hospital.patients[p.id]
                                                self.hospital.releases += 1
                        # --->
                        else:

                            # Length of stay complete. Remove patient from counts and
                            # dictionaries
                            self.hospital.bed_count -= 1
                            # print('Patient %d leaving bed %7.2f, bed count %d' %(p.id,self.env.now,self.hospital.bed_count))
                            # print('Patient %d released %7.2f, bed count %d' %(p.id,self.env.now,self.hospital.bed_count))
                            del self.hospital.patients_in_beds[p.id]
                            # TODO: review bed release
                            self.resources.beds.release(req)
                            del self.hospital.patients[p.id]
                            self.hospital.releases += 1

            # icu bed
            else:

                self.hospital.admissions_icu += 1

                # Increment queue count
                self.hospital.queue_icu_count += 1
                # print('Patient %d arriving icu queue %7.2f, queue icu count %d' %(p.id,self.env.now,self.hospital.queue_icu_count))
                # print('Occupied Beds %d'%(self.hospital.bed_icu_count))

                # Add patient to dictionary of icu queuing patients. This is not used
                # further in this model.
                self.hospital.patients_in_icu_queue[p.id] = p

                # Yield resource request. Sim continues after yield when resources
                # are vailable (so there is no delay if resources are immediately
                # available)
                # TODO: review paciente priority
                with self.resources_icu.icu_beds.request(priority=2) as icu_req:

                    final = yield icu_req | self.env.timeout(p.los_uti)

                    if not (icu_req in final):

                        is_dead = 1 if random.uniform(0, 1) < g.icu_queue_death_rate else 0

                        if is_dead == 1:
                            self.hospital.queue_icu_count -= 1
                            # print('Patient %d leaving icu bed %7.2f, icu bed count %d' %(p.id,self.env.now,self.hospital.bed_icu_count))
                            del self.hospital.patients_in_icu_queue[p.id]
                            del self.hospital.patients[p.id]
                            self.resources_icu.icu_beds.release(icu_req)
                            self.hospital.queue_ICUDeath += 1

                        else:
                            self.hospital.queue_icu_count -= 1
                            # print('Patient %d leaving icu bed %7.2f, icu bed count %d' %(p.id,self.env.now,self.hospital.bed_icu_count))
                            del self.hospital.patients_in_icu_queue[p.id]
                            del self.hospital.patients[p.id]
                            self.resources_icu.icu_beds.release(icu_req)
                            self.hospital.icu_queue_releases += 1

                    else:

                        # Resource now available. Remove from queue count and dictionary of
                        # queued objects
                        self.hospital.queue_icu_count -= 1
                        del self.hospital.patients_in_icu_queue[p.id]
                        # print('Patient %d leaving icu queue %7.2f, icu queue count %d' %(p.id,self.env.now,self.hospital.queue_icu_count))

                        # Add to count of patients in icu beds and to dictionary of patients in
                        # icu beds
                        self.hospital.patients_in_icu_beds[p.id] = p
                        self.hospital.bed_icu_count += 1
                        # print('Patient %d arriving icu bed %7.2f, icu bed count %d' %(p.id,self.env.now,self.hospital.bed_icu_count))

                        # Trigger length of stay delay
                        yield self.env.timeout(p.los_uti)

                        is_dead = 1 if random.uniform(0, 1) < g.icu_death_rate else 0

                        if is_dead == 1:
                            self.hospital.bed_icu_count -= 1
                            # print('Patient %d leaving icu bed %7.2f, icu bed count %d' %(p.id,self.env.now,self.hospital.bed_icu_count))
                            del self.hospital.patients_in_icu_beds[p.id]
                            del self.hospital.patients[p.id]
                            self.resources_icu.icu_beds.release(icu_req)
                            self.hospital.ICUDeath += 1

                        else:

                            # TODO: review pacients going to bed
                            # Goes to bed

                            # Increment queue count
                            self.hospital.queue_count += 1
                            # print('Patient %d arriving queue %7.2f, queue count %d' %(p.id,self.env.now,self.hospital.queue_count))
                            # print('Occupied Beds %d'%(self.hospital.bed_count))

                            # Add patient to dictionary of queuing patients. This is not used
                            # further in this model.
                            self.hospital.patients_in_queue[p.id] = p

                            # Yield resource request. Sim continues after yield when resources
                            # are vailable (so there is no delay if resources are immediately
                            # available)
                            with self.resources.beds.request(priority=1) as req:

                                final = yield req | self.env.timeout(p.los)

                                if not (req in final):

                                    is_dead = 1 if random.uniform(0, 1) < g.queue_death_rate else 0

                                    if is_dead == 1:
                                        self.hospital.queue_count -= 1
                                        # print('Patient %d leaving icu bed %7.2f, icu bed count %d' %(p.id,self.env.now,self.hospital.bed_icu_count))
                                        del self.hospital.patients_in_queue[p.id]
                                        self.resources.beds.release(req)

                                        del self.hospital.patients_in_icu_beds[p.id]
                                        self.hospital.bed_icu_count -= 1

                                        del self.hospital.patients[p.id]
                                        self.hospital.queue_Death += 1

                                    else:
                                        self.hospital.queue_count -= 1
                                        # print('Patient %d leaving icu bed %7.2f, icu bed count %d' %(p.id,self.env.now,self.hospital.bed_icu_count))
                                        del self.hospital.patients_in_queue[p.id]
                                        del self.hospital.patients[p.id]
                                        self.resources.beds.release(req)
                                        self.hospital.queue_releases += 1

                                else:


                                    # Resource now available. Remove from queue count and dictionary of
                                    # queued objects
                                    self.hospital.queue_count -= 1
                                    del self.hospital.patients_in_queue[p.id]
                                    # print('Patient %d leaving queue %7.2f, queue count %d' %(p.id,self.env.now,self.hospital.queue_count))
                                    self.hospital.bed_icu_count -= 1
                                    # print('Patient %d leaving icu bed %7.2f, icu bed count %d' %(p.id,self.env.now,self.hospital.bed_icu_count))
                                    del self.hospital.patients_in_icu_beds[p.id]
                                    self.resources_icu.icu_beds.release(icu_req)
                                    self.hospital.releases_ICU += 1

                                    # TODO: review
                                    # Add to count of patients in beds and to dictionary of patients in
                                    # beds
                                    self.hospital.patients_in_beds[p.id] = p
                                    self.hospital.bed_count += 1
                                    # print('Patient %d arriving bed %7.2f, bed count %d' %(p.id,self.env.now,self.hospital.bed_count))

                                    self.hospital.ICUToBed += 1

                                    # Trigger length of stay delay
                                    yield self.env.timeout(p.los)

                                    # Length of stay complete. Remove patient from counts and
                                    # dictionaries
                                    self.hospital.bed_count -= 1
                                    print('Patient %d leaving bed %7.2f, bed count %d' % (
                                    p.id, self.env.now, self.hospital.bed_count))
                                    print('Patient %d released %7.2f, bed count %d' % (
                                    p.id, self.env.now, self.hospital.bed_count))
                                    del self.hospital.patients_in_beds[p.id]
                                    self.resources.beds.release(req)
                                    del self.hospital.patients[p.id]
                                    self.hospital.releases += 1

            return

        def run(self):
            """
            Controls the main model run. Initialises model and patient arrival and
            audit processes. Instigates the run. At end of run calls for an audit
            summary and bed occupancy plot
            """

            # Set up hospital (calling Hospital class)
            self.hospital = Hospital()

            # Set up resources (beds, icu_beds)
            self.resources = Resources(self.env, g.beds)
            self.resources_icu = Resources_ICU(self.env, g.icu_beds)

            # Set up starting processes: new admissions and bed  audit (with delay)
            self.env.process(self.new_admission(g.inter_arrival_time))
            self.env.process(self.audit_beds(delay=1))
            halt_process = self.env.process(self.halt(delay=3))

            # Start model run
            self.env.run(until=halt_process)

            # At end of run call for bed audit summary and bed occupancy plot
            self.hospital.build_audit_report()
            # self.hospital.chart()

            return

    # In[22]:
    class Patient:
        """
        Patient class. Contains patient id and length of stay (it could contain
        other info about patient, such as priority or clinical group.
        The only method is __init__ for creating a patient (with assignment of
        patient id and length of stay).
        """

        def __init__(self, patient_id, los, los_uti):
            """
            Contructor for new patient.
            :param patient_id: id of patient  (set in self.new_admission)
            :param los: length of stay (days, set in self.new_admission)
            """
            self.id = patient_id
            self.los = los
            self.los_uti = los_uti

            return

    # In[23]:
    class Resources:
        """
        Holds beds resources
        """

        def __init__(self, env, number_of_beds):
            """        Constructor method to initialise beds resource)"""
            self.beds = simpy.PriorityResource(env, capacity=number_of_beds)

            return

    # In[24]:
    class Resources_ICU:
        """
        Holds icu beds resources
        """

        def __init__(self, env, number_of_icu_beds):
            """        Constructor method to initialise icu beds resource)"""
            self.icu_beds = simpy.PriorityResource(env, capacity=number_of_icu_beds)

            return

    seed(98989)
    model = Model()
    model.run()
    return model.hospital.build_audit_report()

