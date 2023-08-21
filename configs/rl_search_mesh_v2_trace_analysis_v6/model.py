import model_Simple_FC1

def config_policy_model(arg, state_dim, action_d_dim, action_p_dim):

	arch = arg.split('_')
	if   (arch[0] == 'SimpleFC1'):
		model = model_Simple_FC1.Simple_FC1(state_dim, action_d_dim, action_p_dim)
	else:
		print('[ERRO] Unspecified model')
		assert(0)

	return model