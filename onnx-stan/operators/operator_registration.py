from onnxruntime import onnx, helper, AttributeProto, TensorProto, GraphProto
from onnx import defs, helper, checker
from onnxruntime_extensions import get_library_path, PyOp

from scipy.special import logsumexp

import numpy as np
import scipy.stats as stats


def register_custom_ops():
    # Custom operator schemas
    operator_schemas = [
        helper.make_opsetid(domain='com.custom', version=1)
    ]

    # Gradient of target log probability
    gradient_op_schema = onnx.OperatorSchema('GradientOfTargetLogProb')
    gradient_op_schema.since_version = 1
    gradient_op_schema.domain = 'com.custom'
    operator_schemas.append(gradient_op_schema)

    # Proposal generation
    proposal_op_schema = onnx.OperatorSchema('ProposalGeneration')
    proposal_op_schema.since_version = 1
    proposal_op_schema.domain = 'com.custom'
    operator_schemas.append(proposal_op_schema)

    # Metropolis-Hastings acceptance
    mh_op_schema = onnx.OperatorSchema('MetropolisHastingsAcceptance')
    mh_op_schema.since_version = 1
    mh_op_schema.domain = 'com.custom'
    operator_schemas.append(mh_op_schema)

    # Leapfrog integration
    leapfrog_op_schema = onnx.OperatorSchema('LeapfrogIntegration')
    leapfrog_op_schema.since_version = 1
    leapfrog_op_schema.domain = 'com.custom'
    operator_schemas.append(leapfrog_op_schema)

    # Pathfinder algorithm
    pathfinder_op_schema = onnx.OperatorSchema('PathFinder')
    pathfinder_op_schema.since_version = 1
    pathfinder_op_schema.domain = 'com.custom'
    operator_schemas.append(pathfinder_op_schema)

    # Register the custom operator schemas
    onnx.register_op_schema(operator_schemas)

register_custom_ops()


def register_gmm_latent_op():
    # Define the operator schema
    schema = defs.OpSchema(
        'GMMLatent',
        domain='ai.stan',
        since_version=1,
        doc='Compute the PDF of a Gaussian Mixture Model with latent variables',
    )

    # Add inputs and outputs
    schema.input('X', 'Input tensor', 'T')
    schema.input('Weights', 'Mixture weights', 'T')
    schema.input('Means', 'Means of the Gaussians', 'T')
    schema.input('Variances', 'Variances of the Gaussians', 'T')
    schema.output('PDF', 'Probability density function at X', 'T')

    # Define the type constraints
    schema.type_constraint('T', ['tensor(float32)', 'tensor(float64)'], 'Constrain input and output types to float tensors')

    # Register the operator schema
    defs.onnx_opset_version = 13  # Update this to the ONNX version you're using
    defs.add(schema)

register_gmm_latent_op()


class NestedHierarchicalNormal(PyOp):
    @staticmethod
    def forward(x, hyperhyperparameters):
        # Unpack the hyperhyperparameters
        mu_mu, mu_sigma, sigma_mu, sigma_sigma = hyperhyperparameters

        # Sample the hyperparameters from the hyperpriors
        mu = np.random.normal(mu_mu, mu_sigma)
        sigma = np.random.normal(sigma_mu, sigma_sigma)

        # Compute the PDF of the normal distribution with the sampled hyperparameters
        pdf = stats.norm.pdf(x, loc=mu, scale=sigma)

        return pdf


def register_nested_hierarchical_op():
    # Define the operator schema
    schema = defs.OpSchema(
        'NestedHierarchicalNormal',
        domain='ai.stan',
        since_version=1,
        doc='A two-level hierarchical Normal-Normal model',
    )

    # Add inputs and outputs
    schema.input('X', 'Input tensor', 'T')
    schema.input('HyperHyperparameters', 'HyperHyperparameters tensor', 'T')
    schema.output('PDF', 'Probability density function at X', 'T')

    # Define the type constraints
    schema.type_constraint('T', ['tensor(float32)', 'tensor(float64)'], 'Constrain input and output types to float tensors')

    # Register the operator schema
    defs.onnx_opset_version = 13  # Update this to the ONNX version you're using
    defs.add(schema)

register_nested_hierarchical_op()


class HierarchicalNormal(PyOp):
    @staticmethod
    def forward(x, hyperparameters):
        # Unpack the hyperparameters
        mu, sigma, tau = hyperparameters

        # Sample the mean from the hyperprior
        mean = np.random.normal(mu, sigma)

        # Compute the PDF of the normal distribution with the sampled mean
        pdf = stats.norm.pdf(x, loc=mean, scale=tau)

        return pdf


def register_hierarchical_op():
    # Define the operator schema
    schema = defs.OpSchema(
        'HierarchicalNormal',
        domain='ai.stan',
        since_version=1,
        doc='A hierarchical Normal-Normal model',
    )

    # Add inputs and outputs
    schema.input('X', 'Input tensor', 'T')
    schema.input('Hyperparameters', 'Hyperparameters tensor', 'T')
    schema.output('PDF', 'Probability density function at X', 'T')

    # Define the type constraints
    schema.type_constraint('T', ['tensor(float32)', 'tensor(float64)'], 'Constrain input and output types to float tensors')

    # Register the operator schema
    defs.onnx_opset_version = 13  # Update this to the ONNX version you're using
    defs.add(schema)

register_hierarchical_op()


class GMM(PyOp):
    @staticmethod
    def forward(x, weights, means, variances):
        # Compute the log PDF of each Gaussian
        logpdfs = [stats.norm.logpdf(x, loc=mean, scale=np.sqrt(var)) for mean, var in zip(means, variances)]

        # Compute the weighted sum in log space
        weighted_logpdfs = np.log(weights) + logpdfs

        # Compute the log PDF of the GMM
        logpdf = logsumexp(weighted_logpdfs, axis=0)

        return logpdf


def register_gmm_op():
    # Define the operator schema
    schema = defs.OpSchema(
        'GMM',
        domain='ai.stan',
        since_version=1,
        doc='Compute the log PDF of a Gaussian Mixture Model',
    )

    # Add inputs and outputs
    schema.input('X', 'Input tensor', 'T')
    schema.input('Weights', 'Mixture weights', 'T')
    schema.input('Means', 'Means of the Gaussians', 'T')
    schema.input('Variances', 'Variances of the Gaussians', 'T')
    schema.output('LogPDF', 'Log of the probability density function at X', 'T')

    # Define the type constraints
    schema.type_constraint('T', ['tensor(float32)', 'tensor(float64)'], 'Constrain input and output types to float tensors')

    # Register the operator schema
    defs.onnx_opset_version = 13  # Update this to the ONNX version you're using
    defs.add(schema)

register_gmm_op()


class WelfordVariance(PyOp):
    @staticmethod
    def forward(x):
        mean = np.mean(x)
        s = np.sum((x - mean)**2)
        return s / (len(x) - 1)


def register_welford_op():
    # Define the operator schema
    schema = defs.OpSchema(
        'WelfordVariance',
        domain='ai.stan',
        since_version=1,
        doc='Compute the variance of a tensor using Welford\'s method',
    )

    # Add inputs and outputs
    schema.input('X', 'Input tensor', 'T')
    schema.output('Variance', 'Variance of the elements in X', 'T')

    # Define the type constraints
    schema.type_constraint('T', ['tensor(float32)', 'tensor(float64)'], 'Constrain input and output types to float tensors')

    # Register the operator schema
    defs.onnx_opset_version = 13  # Update this to the ONNX version you're using
    defs.add(schema)

register_welford_op()


class Gamma(PyOp):
    @staticmethod
    def forward(x, shape, scale):
        # Check for invalid parameters
        if np.any(shape <= 0) or np.any(scale <= 0):
            return np.full_like(x, np.nan)

        # Compute the PDF of the Gamma distribution
        return stats.gamma.pdf(x, a=shape, scale=scale)


def register_gamma_op():
    # Define the operator schema
    schema = defs.OpSchema(
        'Gamma',
        domain='ai.stan',
        since_version=1,
        doc='Compute the PDF of a Gamma distribution',
    )

    # Add inputs and outputs
    schema.input('X', 'Input tensor', 'T')
    schema.input('Shape', 'Shape parameter', 'T')
    schema.input('Scale', 'Scale parameter', 'T')
    schema.output('PDF', 'Probability density function at X', 'T')

    # Define the type constraints
    schema.type_constraint('T', ['tensor(float32)', 'tensor(float64)'], 'Constrain input and output types to float tensors')

    # Register the operator schema
    defs.onnx_opset_version = 13  # Update this to the ONNX version you're using
    defs.add(schema)

register_gamma_op()


class LogNormal(PyOp):
    @staticmethod
    def forward(x):
        # Compute the log PDF of the normal distribution
        return stats.norm.logpdf(x)


def register_lognormal_op():
    # Define the operator schema
    schema = defs.OpSchema(
        'LogNormal',
        domain='ai.stan',
        since_version=1,
        doc='Compute the log PDF of a normal distribution',
    )

    # Add inputs and outputs
    schema.input('X', 'Input tensor', 'T')
    schema.output('LogPDF', 'Log of the probability density function at X', 'T')

    # Define the type constraints
    schema.type_constraint('T', ['tensor(float32)', 'tensor(float64)'], 'Constrain input and output types to float tensors')

    # Register the operator schema
    defs.onnx_opset_version = 13  # Update this to the ONNX version you're using
    defs.add(schema)

register_lognormal_op()


class MixtureGaussian(PyOp):
    @staticmethod
    def forward(x, weights, means, variances):
        # Compute the PDF of the mixture of Gaussians distribution
        pdf = np.sum([w * stats.norm.pdf(x, loc=m, scale=np.sqrt(v)) for w, m, v in zip(weights, means, variances)], axis=0)
        return pdf


def register_mixture_op():
    # Define the operator schema
    schema = defs.OpSchema(
        'MixtureGaussian',
        domain='ai.stan',
        since_version=1,
        doc='Compute the PDF of a mixture of Gaussians distribution',
    )

    # Add inputs and outputs
    schema.input('X', 'Input tensor', 'T')
    schema.input('Weights', 'Mixture weights', 'T')
    schema.input('Means', 'Means of the Gaussians', 'T')
    schema.input('Variances', 'Variances of the Gaussians', 'T')
    schema.output('PDF', 'Probability density function at X', 'T')

    # Define the type constraints
    schema.type_constraint('T', ['tensor(float32)', 'tensor(float64)'], 'Constrain input and output types to float tensors')

    # Register the operator schema
    defs.onnx_opset_version = 13  # Update this to the ONNX version you're using
    defs.add(schema)

register_mixture_op()


class Bernoulli(PyOp):
    @staticmethod
    def forward(x):
        # Compute the PMF of the Bernoulli distribution
        return stats.bernoulli.pmf(x)


def register_bernoulli_op():
    # Define the operator schema
    schema = defs.OpSchema(
        'Bernoulli',
        domain='ai.stan',
        since_version=1,
        doc='Compute the PMF of a Bernoulli distribution',
    )

    # Add inputs and outputs
    schema.input('X', 'Input tensor', 'T')
    schema.output('PMF', 'Probability mass function at X', 'T')

    # Define the type constraints
    schema.type_constraint('T', ['tensor(float32)', 'tensor(float64)'], 'Constrain input and output types to float tensors')

    # Register the operator schema
    defs.onnx_opset_version = 13  # Update this to the ONNX version you're using
    defs.add(schema)

register_bernoulli_op()


class Normal(PyOp):
    @staticmethod
    def forward(x):
        # Compute the PDF of the normal distribution
        return stats.norm.pdf(x)


def register_normal_op():
    # Define the operator schema
    schema = defs.OpSchema(
        'Normal',
        domain='ai.stan',
        since_version=1,
        doc='Compute the PDF of a normal distribution',
    )

    # Add inputs and outputs
    schema.input('X', 'Input tensor', 'T')
    schema.output('PDF', 'Probability density function at X', 'T')

    # Define the type constraints
    schema.type_constraint('T', ['tensor(float32)', 'tensor(float64)'], 'Constrain input and output types to float tensors')

    # Register the operator schema
    defs.onnx_opset_version = 13  # Update this to the ONNX version you're using
    defs.add(schema)

register_normal_op()
