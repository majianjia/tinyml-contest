import os

#if(not os.path.exists('CMSIS_5')):
#    os.system('git clone https://github.com/ARM-software/CMSIS_5.git')

ROOT=os.path.abspath('nnom')

env = Environment()
env.Replace(
    ARCOMSTR = 'AR $SOURCE',
    ASCOMSTR = 'AS $SOURCE',
    ASPPCOMSTR = 'AS $SOURCE',
    CCCOMSTR = 'CC $SOURCE',
    CXXCOMSTR = 'CXX $SOURCE',
    LINKCOMSTR = 'LINK $TARGET'
)

objs = []

#objs += Glob('CMSIS_5/CMSIS/NN/Source/*/*.c')
#objs += Glob('CMSIS_5/CMSIS/DSP/Source/BasicMathFunctions/arm_*.c')
objs += Glob('main.c')

#env.Append(CPPPATH=['CMSIS_5/CMSIS/NN/Include',
#                      'CMSIS_5/CMSIS/DSP/Include',
#                      'CMSIS_5/CMSIS/Core/Include'])
#env.Append(CPPDEFINES=['__ARM_ARCH_8M_BASE__'])
#env.Append(CCFLAGS=['-g','-O0','-std=gnu99'])
env.Append(CCFLAGS=['-std=c99'])

objs +=Glob('%s/src/core/*.c'%(ROOT))
objs +=Glob('%s/src/layers/*.c'%(ROOT))
objs +=Glob('%s/src/backends/*.c'%(ROOT))
env.Append(CPPPATH=['%s/inc'%(ROOT),'%s/port'%(ROOT)])
#env.Append(CPPDEFINES=['USE_NNOM_OUTPUT_SAVE'])

#if(os.getenv('USE_CMSIS_NN') == 'YES'):
#	env.Append(CPPDEFINES=['NNOM_USING_CMSIS_NN'])

env.Program('mnist',objs)