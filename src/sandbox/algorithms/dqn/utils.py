import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return (patch,)


def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis("off")
    anim = animation.FuncAnimation(
        fig,
        update_scene,
        fargs=(frames, patch),
        frames=len(frames),
        repeat=repeat,
        interval=interval,
    )
    plt.show()


def enjoy(env, policy) -> None:
    state = env.reset()

    frames = []

    for step in range(200):
        action = policy(state)
        state, reward, done, info = env.step(action)
        if done:
            break
        img = env.render(mode="rgb_array")
        frames.append(img)

    plot_animation(frames)