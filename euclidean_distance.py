def euclideanMetric(f_component, s_component):
    if len(f_component) != len(s_component):
        raise '[+] Both points must be in the same dimension!'
    
    d = sum([(s_component[k] - f_component[k])**2 for k in range(len(f_component))])
    
    return d**(1/2)