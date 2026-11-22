"""Original ceramic/flower feedback on the existing mixer. Generated waveforms: CC0-1.0."""
import numpy as np
import pygame


def cue_wave(cue,rate):
    duration={'shatter':.42,'topple':.32,'petal':.24}[cue]
    t=np.arange(round(rate*duration),dtype=np.float64)/rate
    rng=np.random.default_rng({'shatter':271,'topple':272,'petal':273}[cue])
    if cue=='shatter':
        noise=rng.standard_normal(len(t));noise-=np.convolve(noise,np.ones(9)/9,mode='same')
        wave=.45*noise*np.exp(-18*t)
        for delay,freq in [(0,1150),(.055,1630),(.11,2120),(.18,1350),(.25,1900)]:
            u=np.maximum(0,t-delay);wave+=np.where(t>=delay,.32*np.sin(2*np.pi*freq*u)*np.exp(-35*u),0)
    elif cue=='topple':
        wave=np.sin(2*np.pi*350*t)*np.exp(-22*t)
        u=np.maximum(0,t-.13);wave+=np.where(t>=.13,.5*np.sin(2*np.pi*920*u)*np.exp(-28*u),0)
    else:
        wave=np.sin(2*np.pi*(740*t-650*t*t))*np.exp(-16*t)
    # Quiet feedback; digital level is not an acoustic calibration or input veto.
    wave*=.16/max(float(np.max(np.abs(wave))),1e-9)
    wave*=np.minimum(1,t/.003)*np.minimum(1,(duration-t)/.015)
    return np.asarray(wave*32767,dtype=np.int16)


def build_sounds():
    config=pygame.mixer.get_init()
    if not config or config[1]!=-16 or config[2] not in (1,2):return {}
    sounds={}
    try:
        rate,_,channels=config
        for cue in ('shatter','topple','petal'):
            wave=cue_wave(cue,rate)
            if channels==2:wave=np.repeat(wave[:,None],2,axis=1)
            sounds[cue]=pygame.sndarray.make_sound(np.ascontiguousarray(wave))
    except (pygame.error,RuntimeError):return {}
    return sounds
