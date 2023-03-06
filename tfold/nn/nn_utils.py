def generate_registers_I(l):
    '''
    generate all admissible pairs of tails for a cl II pep of length l;
    assume only combinations (0,x) and (x,0) allowed, to avoid quadratic proliferaton of registers;
    experimentally, there is only one structure with tails (-1,1) violating this rule
    '''
    if l<8:
        raise ValueError('peplen<8 not allowed for cl I')
    registers=[]
    for i in range(-1,min(2,l-7)):
        registers.append((i,0))
    if l>8:
        for i in range(1,l-7):
            registers.append((0,i))
    return registers
def generate_registers_II(l):
    '''
    generate all admissible pairs of tails for a cl II pep of length l;
    assume a flat binding core of len 9aa;
    experimentally, there is one structure with peplen 10 and left tail -1 violating the rule,
    and no structures with bulged or stretched binding core, except possibly for pig MHC or such
    '''
    if l<9:
        raise ValueError('peplen<9 not allowed for cl II')    
    registers=[]
    for i in range(l-8):
        registers.append((i,l-i-9))
    return registers
    