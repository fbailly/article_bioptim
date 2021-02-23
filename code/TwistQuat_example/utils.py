import biorbd
import casadi as cas
from bioptim import PenaltyNodes

def states2eulerRate(states):
    # maximizing Lagrange twist velocity (indetermination of Quaternion to Euler of 2*pi*n)

    def bodyVel2eulerRate(w, e):
        # xyz convention
        ph = e[0]
        th = e[1]
        ps = e[2]
        wx = w[0]
        wy = w[1]
        wz = w[2]
        dph = cas.cos(ps) / cas.cos(th) * wx - cas.sin(ps) / cas.cos(th) * wy
        dth = cas.sin(ps) * wx + cas.cos(ps) * wy
        dps = -cas.cos(ps) * cas.sin(th) / cas.cos(th) * wx + cas.sin(th) * cas.sin(ps) / cas.cos(th) * wy + wz
        return cas.vertcat(dph, dth, dps)

    Quat_cas = cas.vertcat(states[8], states[3], states[4], states[5])
    Quat_cas /= cas.norm_fro(Quat_cas)

    Quaterion = biorbd.Quaternion(Quat_cas[0], Quat_cas[1], Quat_cas[2], Quat_cas[3])

    omega = cas.vertcat(states[12:15])
    euler = biorbd.Rotation.toEulerAngles(biorbd.Quaternion.toMatrix(Quaterion), 'xyz').to_mx()
    EulRate = bodyVel2eulerRate(omega, euler)

    Func = cas.Function('MaxTwist', [states], [EulRate])
    return Func


def MaxTwistQuat(pn: PenaltyNodes, states2eulerRate_func) -> cas.MX:
    val = []
    for i in range(pn.nlp.ns):
        val = cas.vertcat(val, states2eulerRate_func(pn.x[i])[-1])
    return val


def states2euler(states):
    Quat_cas = cas.vertcat(states[8], states[3], states[4], states[5])
    Quat_cas /= cas.norm_fro(Quat_cas)
    Quaterion = biorbd.Quaternion(Quat_cas[0], Quat_cas[1], Quat_cas[2], Quat_cas[3])
    euler = biorbd.Rotation.toEulerAngles(biorbd.Quaternion.toMatrix(Quaterion), 'xyz').to_mx()
    Func = cas.Function('states2euler', [states], [euler])
    return Func


def FinalPositionQuat(pn: PenaltyNodes, states2euler_func) -> cas.MX:
    val = states2euler_func(pn.x[0])[0]
    return val


def Eul2Quat(Eul):
    Quat = biorbd.Quaternion.fromMatrix(biorbd.Rotation.fromEulerAngles(Eul, 'xyz')).to_mx()
    Quat /= cas.norm_fro(Quat)
    Func = cas.Function('Eul2Quat', [Eul], [Quat])
    return Func


def EulRate2BodyVel(Quat, EulRate, Eul):
    Quat_biorbd = biorbd.Quaternion(Quat[0], Quat[1], Quat[2], Quat[3])
    EulRate_biorbd = biorbd.Vector3d(EulRate[0], EulRate[1], EulRate[2])
    Eul_biorbd = biorbd.Vector3d(Eul[0], Eul[1], Eul[2])
    Omega = biorbd.Quaternion.eulerDotToOmega(Quat_biorbd, EulRate_biorbd, Eul_biorbd, 'xyz').to_mx()
    Func = cas.Function('Eul2Quat', [Quat, EulRate, Eul], [Omega])
    return Func


