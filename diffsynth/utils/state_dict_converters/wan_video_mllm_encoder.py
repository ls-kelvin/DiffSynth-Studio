def WanMLLMEncoderStateDictConverter(state_dict):
    state_dict_ = {}
    for k in state_dict:
        v = state_dict[k]
        state_dict_[k] = v
    return state_dict_
