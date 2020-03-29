import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma


class BasicReproductionNumber:

    def __init__(self, incidence, prior_shape=1, prior_scale=5,
                 si_pmf=None, si_pars=None, t_start=None, window_width=6):
        """Initialize ReproductionNumber class

        Args:
            incidence(DataFrame): must have columns 'dates' and 
                'incidence' (number of new cases per day).
            prior_shape (float): value of shape parameter of Gamma prior for
                reproduction number estimation.
            prior_scale (float): value of scale parameter of Gamma prior for
                reproduction number estimation.
            si_pmf (DataFrame): must have columns 'interval_length' and 
                'probability'.Represents probability mass function for given 
                values of serial interval.
            si_pars (dict): dictionary with keys 'mean' and 'sd'. 
                Represents parameters to generate PMF for serial interval.

        Notes:
            You must specify at least one of si_pmf or si_pars, but not both.

        """

        self.incidence = incidence
        self.prior_shape = prior_shape
        self.prior_scale = prior_scale
        self.si_pmf = si_pmf
        self.si_pars = si_pars
        self.t_start = t_start
        self.window_width = window_width
        self.t_end = None
        self.posterior_parameters = {}
        self.posterior_summary = None
        self._check_time_period()
        self._check_serial_number_pmf()

    def _check_time_period(self):
        if self.t_start is None:
            self.t_start = np.arange(1, self.incidence.shape[0] - self.window_width)
        elif isinstance(self.t_start, list):
            self.t_start = np.array(self.t_start)
        self.t_end = self.t_start + self.window_width

    def _check_serial_number_pmf(self):
        if self.si_pmf is not None and self.si_pars is not None:
            txt = "You must pass either 'si_pmf' or 'si_pars', not both."
            raise AttributeError(txt)
        if self.si_pmf is None:
            if self.si_pars is None:
                txt = "You must pass either 'si_pmf' or 'si_pars'. You've passed neither."
                raise AttributeError(txt)
            if not all([i in self.si_pars.keys() for i in ['mean', 'sd']]):
                txt = "'si_pars' must be a dictionary with 'mean' and 'sd' keys."
                raise AttributeError(txt)
            self.compute_serial_interval_pmf()
        else:
            self.si_pmf = self.si_pmf.reset_index().set_index('interval_length')['probability']

    def compute_serial_interval_pmf(self, k=None, mu=None, sigma=None):

        if k is None:
            k = np.arange(self.incidence.shape[0])
        elif not isinstance(k, np.ndarray):
            raise TypeError("k must be of type numpy.ndarray, probably shape = (n_time_windows, ).")

        if mu is None:
            mu = self.si_pars['mean']
        if sigma is None:
            sigma = self.si_pars['sd']

        if sigma < 0:
            raise AttributeError("sigma must be >=0.")
        if mu <= 1:
            raise AttributeError("mu must be >1")
        if not (k >= 0.).sum() == len(k):
            raise AttributeError("all values in k must be >=0.")

        shape = ((mu - 1) / sigma) ** 2
        scale = (sigma ** 2) / (mu - 1)

        def cdf_gamma(x, shape_, scale_):
            return gamma.cdf(x=x, a=shape_, scale=scale_)

        si_pmf = (k*cdf_gamma(k, shape, scale) 
                  + (k - 2)*cdf_gamma(k - 2, shape, scale) 
                  - 2*(k - 1)*cdf_gamma(k - 1, shape, scale))
        si_pmf = si_pmf + shape*scale*(2 * cdf_gamma(k - 1, shape + 1, scale) 
                                       - cdf_gamma(k - 2, shape + 1, scale) 
                                       - cdf_gamma(k, shape + 1, scale))
        si_pmf = np.array([np.max([0, i]) for i in si_pmf])

        self.si_pmf = si_pmf

    def compute_overall_infectivity(self):

        def fill_up_with_zeros(x, ref):
            x_nrows, ref_nrows = x.shape[0], ref.shape[0]
            updated_x = x
            if x_nrows < ref_nrows:
                updated_x = np.concatenate([x, np.zeros(1 + ref_nrows - x_nrows)])
            return updated_x

        incid, si_pmf = self.incidence, self.si_pmf
        si_pmf = fill_up_with_zeros(x=si_pmf, ref=incid)
        number_of_time_points = incid.shape[0]
        overall_infectivity = np.zeros((number_of_time_points,))
        for t in range(1, number_of_time_points + 1):
            overall_infectivity[t - 1] = (si_pmf[:t] * incid.iloc[:t][::-1]['incidence']).sum()
        overall_infectivity[0] = np.nan

        return overall_infectivity

    def compute_posterior_parameters(self, prior_shape=None, prior_scale=None):
        incid, si_pmf = self.incidence, self.si_pmf
        t_start, t_end = self.t_start, self.t_end
        if prior_shape is None:
            prior_shape = self.prior_shape
        if prior_scale is None:
            prior_scale = self.prior_scale

        number_of_time_windows = len(t_start)
        overall_infectivity = self.compute_overall_infectivity()
        final_mean_si = (si_pmf * range(len(si_pmf))).sum()

        posterior_shape = np.zeros(number_of_time_windows)
        posterior_scale = np.zeros(number_of_time_windows)

        for t in range(number_of_time_windows):
            if t_end[t] > final_mean_si:
                posterior_shape[t] = prior_shape + (incid.iloc[range(t_start[t], t_end[t] + 1)]["incidence"]).sum()
            else:
                posterior_shape[t] = np.nan

        for t in range(number_of_time_windows):
            if t_end[t] > final_mean_si:
                period_overall_infectivity = (overall_infectivity[range(t_start[t], t_end[t] + 1)]).sum()
                posterior_scale[t] = 1 / ((1 / prior_scale) + period_overall_infectivity)
            else:
                posterior_scale[t] = np.nan

        self.posterior_parameters['shape'] = posterior_shape
        self.posterior_parameters['scale'] = posterior_scale

    def sample_from_posterior(self, sample_size=1000):
        if not all([i in self.posterior_parameters.keys() for i in ['scale', 'shape']]):
            txt = "Can't sample from posterior before computing posterior parameters."
            raise IndexError(txt)
        posterior_shape = self.posterior_parameters['shape']
        posterior_scale = self.posterior_parameters['scale']
        number_of_time_windows = len(self.t_start)
        sample_r_posterior = np.zeros((number_of_time_windows, sample_size))
        for t in range(number_of_time_windows):
            if not t > len(posterior_shape) - 1:
                sample_r_posterior[t, ] = np.random.gamma(shape=posterior_shape[t],
                                                          scale=posterior_scale[t],
                                                          size=sample_size)
            else:
                sample_r_posterior[t,] = np.nan

        return sample_r_posterior.transpose()

    def compute_posterior_summaries(self, posterior_sample, t_max=None):
        start_dates = self.incidence.index[self.t_start]
        end_dates = self.incidence.index[self.t_end]
        post_mean_r = posterior_sample.mean(axis=0)
        post_sd = posterior_sample.std(axis=0)
        post_shape = self.posterior_parameters['shape']
        post_scale = self.posterior_parameters['scale']
        post_upper_quantile_r = np.quantile(posterior_sample, q=0.975, axis=0)
        post_lower_quantile_r = np.quantile(posterior_sample, q=0.025, axis=0)
        summary_dict = {
            'start_dates': start_dates, 'end_dates': end_dates,
            'Rt_mean': post_mean_r, 'Rt_sd': post_sd,
            'Rt_q0.975': post_upper_quantile_r, 'Rt_q0.025': post_lower_quantile_r,
            'Rt_shape': post_shape, 'Rt_scale': post_scale
        }
        posterior_summary = pd.DataFrame(summary_dict)
        posterior_summary['start_dates'] = posterior_summary['start_dates'].astype('datetime64[ns]')
        posterior_summary['end_dates'] = posterior_summary['end_dates'].astype('datetime64[ns]')

        if t_max is not None:
            last_day = max(posterior_summary['end_dates'])
            final_date = max(posterior_summary['end_dates']) + pd.Timedelta(days=t_max)
            last_day_data = posterior_summary[posterior_summary['end_dates'] == last_day].to_dict(orient='list')
            dates_ahead = pd.date_range(start=last_day, end=final_date)[1:]

            forecast_d = pd.DataFrame({
                'start_dates': pd.NaT, 'end_dates': dates_ahead
            })

            forecast_d['Rt_mean'] = last_day_data['Rt_mean'][0]
            forecast_d['Rt_sd'] = last_day_data['Rt_sd'][0]
            forecast_d['Rt_q0.975'] = last_day_data['Rt_q0.975'][0]
            forecast_d['Rt_q0.025'] = last_day_data['Rt_q0.025'][0]
            forecast_d['Rt_shape'] = last_day_data['Rt_shape'][0]
            forecast_d['Rt_scale'] = last_day_data['Rt_scale'][0]

            posterior_summary = pd.concat([posterior_summary, forecast_d], ignore_index=True)
            posterior_summary['estimation_type'] = np.where(posterior_summary['end_dates'] <= last_day,
                                                            'fitted', 'forecasted')

        self.posterior_summary = posterior_summary

    def plot_reproduction_number(self, title=None, filename=None):
        d = self.posterior_summary
        if d is None:
            txt = "You need to compute the summaries for the posterior distribution of Rt."
            raise ValueError(txt)
        if title is None:
            title = "R(t): time-varying reproduction number"
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.plot(d['end_dates'], d['Rt_mean'], color='b')
        plt.plot(d['end_dates'], [1] * len(d['Rt_mean']), color='gray', linestyle='dashed', alpha=0.75)
        plt.fill_between(d['end_dates'],
                         d['Rt_q0.975'],
                         d['Rt_q0.025'],
                         color='b', alpha=0.2)
        plt.title(title)
        plt.suptitle("$P(R_t | Data) \sim Gamma(k_t, \\theta_t)$")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        fig.autofmt_xdate()

        if 'estimation_type' in d.columns:
            plt.axvline(x=max(d[d['estimation_type'] == "fitted"]["end_dates"]),
                        color='gray', linestyle='dashed', alpha=0.75)

        if filename is None:
            plt.show()
        else:
            fig.savefig(filename, dpi=fig.dpi)
            plt.close()
