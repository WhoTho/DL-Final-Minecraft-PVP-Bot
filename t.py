import pygame

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

yaw = 0
pitch = 0
sensitivity = 0.2

clock = pygame.time.Clock()

while True:
    dt = clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            print("Releasing mouse and exiting.")
            pygame.event.set_grab(False)
            pygame.mouse.set_visible(True)

    # RELATIVE mouse delta (this works under WSL2!)
    dx, dy = pygame.mouse.get_rel()
    if pygame.key.get_pressed()[pygame.K_q]:
        break

    # Update camera
    yaw += dx * sensitivity
    pitch += dy * sensitivity

    print(f"yaw={yaw:.2f}, pitch={pitch:.2f}")

    screen.fill((0, 0, 0))
    pygame.display.flip()
