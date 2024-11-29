import numpy as np
import matplotlib.pyplot as plt


# 定义单室模型中的参数和方程
def single_compartment_model(V, t, I, gNa, gK, gL, ENa, EK, EL):
    # 离子通道门控变量计算
    exp_term_m = np.exp(-(V[0] + 40) / 10)
    alpha_m = (0.1 * (V[0] + 40)) / (1 - exp_term_m)
    beta_m = 4 * np.exp(-(V[0] + 65) / 18)
    m = alpha_m / (alpha_m + beta_m)
    dmdt = alpha_m * (1 - m) - beta_m * m
    alpha_h = 0.07 * np.exp(-(V[0] + 65) / 20)
    beta_h = 1 / (1 + np.exp(-(V[0] + 35) / 10))
    h = alpha_h / (alpha_h + beta_h)
    dhdt = alpha_h * (1 - h) - beta_h * h
    exp_term_n = np.exp(-(V[0] + 55) / 10)
    alpha_n = (0.01 * (V[0] + 55)) / (1 - exp_term_n)
    beta_n = 0.125 * np.exp(-(V[0] + 65) / 80)
    n = alpha_n / (alpha_n + beta_n)
    dndt = alpha_n * (1 - n) - beta_n * n
    # 膜电流
    INa = gNa * m ** 3 * h * (V[0] - ENa)
    IK = gK * n ** 4 * (V[0] - EK)
    IL = gL * (V[0] - EL)
    # 膜电位的变化率
    dVdt = (-INa - IK - IL + I) / 1
    return np.array([dVdt, dmdt, dhdt, dndt])


# 数值积分求解
def solve_neuron(t, I, gNa, gK, gL, ENa, EK, EL, V0):
    dt = t[1] - t[0]
    num_steps = len(t)
    V = np.zeros((num_steps, 4))
    V[0] = V0[:4]
    for i in range(num_steps - 1):
        k1 = single_compartment_model(V[i], t, I, gNa, gK, gL, ENa, EK, EL)
        # 避免不必要的数组复制，直接在原数组上操作
        V[i, 0] += k1[0] * dt / 2
        V[i, 1] += k1[1] * dt / 2
        V[i, 2] += k1[2] * dt / 2
        V[i, 3] += k1[3] * dt / 2
        k2 = single_compartment_model(V[i], t + dt / 2, I, gNa, gK, gL, ENa, EK, EL)
        V[i, 0] += k2[0] * dt / 2
        V[i, 1] += k2[1] * dt / 2
        V[i, 2] += k2[2] * dt / 2
        V[i, 3] += k2[3] * dt / 2
        k3 = single_compartment_model(V[i], t + dt / 2, I, gNa, gK, gL, ENa, EK, EL)
        V[i, 0] += k3[0] * dt
        V[i, 1] += k3[1] * dt
        V[i, 2] += k3[2] * dt
        V[i, 3] += k3[3] * dt
        k4 = single_compartment_model(V[i], t + dt, I, gNa, gK, gL, ENa, EK, EL)
        V[i + 1] = V[i] + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    return V


# 电缆方程
def cable_equation(V, t, dx, r_m, r_i, I_ext):
    num_points = len(V)
    # 向量化计算d2Vdx2
    d2Vdx2 = np.roll(V, -1) - 2 * V + np.roll(V, 1)
    d2Vdx2[0] = (V[1] - 2 * V[0]) / dx ** 2
    d2Vdx2[-1] = (V[-2] - 2 * V[-1]) / dx ** 2
    dVdt = (1 / r_m) * (r_i * d2Vdx2 + I_ext)
    return dVdt


# 结合单室模型和电缆方程求解
def solve_neuron_with_cable(gNa, gK, gL=0.3, ENa=50, EK=-77, EL=-54.4, I=10, t_max=100, dt=0.01, dx=0.1, r_m=10,
                            r_i=1):
    t = np.arange(0, t_max, dt)
    num_points = int(1 / dx)
    V = np.zeros((len(t), num_points), dtype=np.float32)
    V[0] = -65

    # 缓存solve_neuron的结果，避免多次调用
    V_temp = solve_neuron(np.arange(0, 100, 0.01), 10, gNa, gK, gL, ENa, EK, EL, np.array([-65, 0, 0, 0]))
    for i in range(len(t) - 1):
        # 先计算单室模型相关的电流部分
        I_ion = np.zeros(num_points)
        for j in range(num_points):
            I_ion[j] = V_temp[-1, 0]

        # 计算电缆方程部分
        dVdt_cable = cable_equation(V[i], t[i], dx, r_m, r_i, I_ion)

        # 整合两部分更新膜电位
        V[i + 1] = V[i] + dVdt_cable * dt
    return t, V


# 绘制正常和癫痫状态下的对比图
def plot_comparison():
    # 正常状态1
    gNa_normal1 = 120
    gK_normal1 = 36
    V1 = solve_neuron(np.arange(0, 100, 0.01), 10, gNa_normal1, gK_normal1, 0.3, 50, -77, -54.4, np.array([-65, 0, 0, 0]))

    # 癫痫状态1
    gNa_epilepsy1 = 360
    gK_epilepsy1 = 36
    V2 = solve_neuron(np.arange(0, 100, 0.01), 10, gNa_epilepsy1, gK_epilepsy1, 0.3, 50, -77, -54.4, np.array([-65, 0, 0, 0]))

    # 考虑空间结构的正常状态
    gNa_normal_cable = 120
    gK_normal_cable = 36
    t5, V5 = solve_neuron_with_cable(gNa_normal_cable, gK_normal_cable)

    # 考虑空间结构的癫痫状态
    gNa_epilepsy_cable = 360
    gK_epilepsy_cable = 36
    t6, V6 = solve_neuron_with_cable(gNa_epilepsy_cable, gK_epilepsy_cable)

  # 绘制动作电位
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.plot(np.arange(0, 100, 0.01), V1[:, 0], label='Normal (GNa = 120, GK = 36)')
    plt.plot(np.arange(0, 100, 0.01), V2[:, 0], label='Epilepsy (GNa = 360, GK = 36)')
    plt.plot(t5, V5[:, int(len(V5[0]) / 2)], label='Normal with Cable (GNa = 120, GK = 36)')
    plt.plot(t6, V6[:, int(len(V6[0]) / 2)], label='Epilepsy with Cable (GNa = 360, GK = 36)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Action Potential')
    plt.legend()

    # 绘制钠电导变化
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(0, 100, 0.01), gNa_normal1 * V1[:, 1] ** 3 * V1[:, 2], label='Normal (GNa = 120, GK = 36)')
    plt.plot(np.arange(0, 100, 0.01), gNa_epilepsy1 * V2[:, 1] ** 3 * V2[:, 2], label='Epilepsy (GNa = 360, GK = 36)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Sodium Conductance (mS/cm²)')
    plt.title('Sodium Conductance Change')
    plt.legend()

    # 绘制钾电导变化
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(0, 100, 0.01), gK_normal1 * V1[:, 3] ** 4, label='Normal (GNa = 120, GK = 36)')
    plt.plot(np.arange(0, 100, 0.01), gK_epilepsy1 * V2[:, 3] ** 4, label='Epilepsy (GNa = 360, GK = 36)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Potassium Conductance (mS/cm²)')
    plt.title('Potassium Conductance Change')
    plt.legend()


    # 绘制两种神经元的空间结构对电信号传播的影响
    plt.subplot(2, 2, 4)
    plt.plot(t5, V5[:, int(len(V5[0]) / 2)], label='Normal with Cable (GNa = 120, GK = 36)')
    plt.plot(t6, V6[:, int(len(V6[0]) / 2)], label='Epilepsy with Cable (GNa = 360, GK = 36)')
    plt.xlabel('Time')
    plt.ylabel('Membrane Potential at Mid - point')
    plt.title('Effect of Spatial Structure on Electrical Signal Propagation')
    plt.legend()
    plt.tight_layout()
    plt.show()
    

    # 正常状态2
    gNa_normal2 = 120
    gK_normal2 = 36
    V3 = solve_neuron(np.arange(0, 100, 0.01), 10, gNa_normal2, gK_normal2, 0.3, 50, -77, -54.4, np.array([-65, 0, 0, 0]))
    # 癫痫状态2
    gNa_epilepsy2 = 120
    gK_epilepsy2 = 7
    V4 = solve_neuron(np.arange(0, 100, 0.01), 10, gNa_epilepsy2, gK_epilepsy2, 0.3, 50, -77, -54.4, np.array([-65, 0, 0, 0]))
    # 考虑空间结构的正常状态
    gNa_normal_cable2 = 120
    gK_normal_cable2 = 36
    t5, V5 = solve_neuron_with_cable(gNa_normal_cable2, gK_normal_cable2)
    # 考虑空间结构的癫痫状态
    gNa_epilepsy_cable2 = 120
    gK_epilepsy_cable2 = 7
    t6, V6 = solve_neuron_with_cable(gNa_epilepsy_cable2, gK_epilepsy_cable2)
    # 绘制动作电位
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(0, 100, 0.01), V3[:, 0], label='Normal (GNa = 120, GK = 36)')
    plt.plot(np.arange(0, 100, 0.01), V4[:, 0], label='Epilepsy (GNa = 120, GK = 7)')
    plt.plot(t5, V5[:, int(len(V5[0]) / 2)], label='Normal with Cable (GNa = 120, GK = 36)')
    plt.plot(t6, V6[:, int(len(V6[0]) / 2)], label='Epilepsy with Cable (GNa = 120, GK = 7)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Action Potential')
    plt.legend()
    # 绘制钠电导变化
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(0, 100, 0.01), gNa_normal2 * V3[:, 1] ** 3 * V3[:, 2], label='Normal (GNa = 120, GK = 36)')
    plt.plot(np.arange(0, 100, 0.01), gNa_epilepsy2 * V4[:, 1] ** 3 * V4[:, 2], label='Epilepsy (GNa = 120, GK = 7)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Sodium Conductance (mS/cm²)')
    plt.title('Sodium Conductance Change')
    plt.legend()
    # 绘制钾电导变化
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(0, 100, 0.01), gK_normal2 * V3[:, 3] ** 4, label='Normal (GNa = 120, GK = 36)')
    plt.plot(np.arange(0, 100, 0.01), gK_epilepsy2 * V4[:, 3] ** 4, label='Epilepsy (GNa = 120, GK = 7)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Potassium Conductance (mS/cm²)')
    plt.title('Potassium Conductance Change')
    plt.legend()
    # 绘制两种神经元的空间结构对电信号传播的影响
    plt.subplot(2, 2, 4)
    plt.plot(t5, V5[:, int(len(V5[0]) / 2)], label='Normal with Cable (GNa = 120, GK = 36)')
    plt.plot(t6, V6[:, int(len(V6[0]) / 2)], label='Epilepsy with Cable (GNa = 120, GK = 7)')
    plt.xlabel('Time')
    plt.ylabel('Membrane Potential at Mid - point')
    plt.title('Effect of Spatial Structure on Electrical Signal Propagation')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_comparison()


