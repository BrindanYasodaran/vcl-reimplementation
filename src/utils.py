import torch

def load_mle_weights(bayesian_model, mle_weights_path):
    INITIAL_RHO = -6.91

    mle_state_dict  = torch.load(mle_weights_path)
    bayesian_state_dict = bayesian_model.state_dict()

    with torch.no_grad():
        for key in mle_state_dict:
            if key.endswith('.weight'):
                param_name = key.replace('.weight', '.weight_mu')
                bayesian_state_dict[param_name].copy_(mle_state_dict[key])
            if key.endswith('.bias'):
                param_name = key.replace('.bias', '.bias_mu')
                bayesian_state_dict[param_name].copy_(mle_state_dict[key])
        
        for key in bayesian_state_dict:
            if "rho" in key:
                bayesian_state_dict[key].fill_(INITIAL_RHO)
    
    bayesian_model.load_state_dict(bayesian_state_dict)


