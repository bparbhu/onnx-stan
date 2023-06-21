#include "onnxruntime/core/graph/contrib_ops/contrib_defs.h"

namespace onnxruntime {
namespace contrib {

ONNX_CONTRIB_OPERATOR_SCHEMA(GradientOfTargetLogProb)
    .SetDomain(kMSDomain)
    .SinceVersion(1)
    .SetDoc("Compute the gradient of target log-probability function w.r.t. the parameters.")
    .Input(0, "parameters", "Current parameters", "T")
    .Output(0, "gradient", "Gradient of the target log-probability function", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_CONTRIB_OPERATOR_SCHEMA(ProposalGeneration)
    .SetDomain(kMSDomain)
    .SinceVersion(1)
    .SetDoc("Generate a proposed new state for HMC.")
    .Input(0, "current_state", "Current state of the Markov chain.", "T")
    .Output(0, "proposed_state", "Proposed new state.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_CONTRIB_OPERATOR_SCHEMA(MetropolisHastingsAcceptance)
    .SetDomain(kMSDomain)
    .SinceVersion(1)
    .SetDoc("Decide whether to accept the proposed new state.")
    .Input(0, "current_state", "Current state of the Markov chain.", "T")
    .Input(1, "proposed_state", "Proposed new state.", "T")
    .Output(0, "new_state", "New state of the Markov chain (either current_state or proposed_state).", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_CONTRIB_OPERATOR_SCHEMA(LeapfrogIntegration)
    .SetDomain(kMSDomain)
    .SinceVersion(1)
    .SetDoc("Perform leapfrog integration to simulate Hamiltonian dynamics.")
    .Input(0, "current_position", "Current position in phase space.", "T")
    .Input(1, "current_momentum", "Current momentum in phase space.", "T")
    .Output(0, "new_position", "New position in phase space.", "T")
    .Output(1, "new_momentum", "New momentum in phase space.", "T")
    .TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");

ONNX_CONTRIB_OPERATOR_SCHEMA(PathFinder)
    .SetDomain(kMSDomain)
    .SinceVersion(1)
    .SetDoc("Find the shortest path in a graph using the Pathfinder algorithm.")
    .Input(0, "graph", "The adjacency matrix representation of the graph.", "T")
    .Input(1, "start_node", "The starting node.", "T")
    .Input(2, "end_node", "The ending node.", "T")
    .Output(0, "shortest_path", "The shortest path from start_node to end_node.", "T")
    .TypeConstraint(
        "T",
        {"tensor(int32)", "tensor(int64)"},
        "Constrain input and output types to integral tensors.");
        
}  // namespace contrib
}  // namespace onnxruntime

