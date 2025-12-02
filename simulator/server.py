# simulator/server.py

import math
from typing import Dict, Optional, List, Any
from simulator.objects import Entity
from simulator import physics, combat
from helpers import vec3


class MinecraftSimulationServer:
    """Minecraft-like PvP simulation server using the new simulator code"""

    def __init__(self):
        self.entities: Dict[int, Entity] = {}
        self.entity_inputs: Dict[int, physics.InputState] = {}
        self.tick_count = 0

        # Minecraft tick rate is 20 TPS
        self.ticks_per_second = 20
        self.tick_duration = 1.0 / self.ticks_per_second

    def add_entity(
        self,
        entity_id: int,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
        pitch: float = 0.0,
        color=(0, 255, 0),
    ) -> Entity:
        """Add a new entity to the simulation"""
        position = vec3.from_list([x, y, z])
        velocity = vec3.zero()

        entity = Entity(
            object_id=entity_id,
            position=position,
            velocity=velocity,
            yaw=yaw,
            pitch=pitch,
            color=color,
        )

        self.entities[entity_id] = entity
        self.entity_inputs[entity_id] = physics.InputState()
        return entity

    def remove_entity(self, entity_id: int) -> bool:
        """Remove an entity from the simulation"""
        if entity_id in self.entities:
            del self.entities[entity_id]
            del self.entity_inputs[entity_id]
            return True
        return False

    def get_entity(self, entity_id: int) -> Optional[Entity]:
        """Get an entity by ID"""
        return self.entities.get(entity_id)

    def take_input(self, entity_id: int, input_map: Dict[str, Any]) -> bool:
        """Update entity input state"""
        if entity_id not in self.entity_inputs:
            return False

        input_state = self.entity_inputs[entity_id]
        entity = self.entities[entity_id]

        # Update input state
        input_state.w = input_map.get("w", input_state.w)
        input_state.a = input_map.get("a", input_state.a)
        input_state.s = input_map.get("s", input_state.s)
        input_state.d = input_map.get("d", input_state.d)
        input_state.sprint = input_map.get("sprint", input_state.sprint)
        input_state.space = input_map.get("space", input_state.space)
        input_state.click = input_map.get("click", input_state.click)
        # TODO: maybe default to current yaw/pitch?
        input_state.yaw = input_map.get("yaw", input_state.yaw)
        input_state.pitch = input_map.get("pitch", input_state.pitch)

        return True

    def step(self):
        """Advance simulation by one tick (1/20th second)"""
        self.tick_count += 1

        # Update all entity timers first
        for entity in self.entities.values():
            if entity.invulnerablility_ticks > 0:
                entity.invulnerablility_ticks -= 1

        # Handle combat for entities that clicked
        entity_list = list(self.entities.values())
        for entity_id, entity in self.entities.items():
            input_state = self.entity_inputs[entity_id]
            if input_state.click:
                combat.try_attack(entity, entity_list)

        # Process physics and movement for all entities
        for entity_id, entity in self.entities.items():
            input_state = self.entity_inputs[entity_id]

            # Apply physics simulation
            physics.simulate(entity, input_state)

        # Prepare output state (could be expanded as needed)
        for input_state in self.entity_inputs.values():
            input_state.click = False  # reset click after processing

    def get_entity_info(self, entity_id: int) -> Optional[Dict]:
        """Get detailed info about an entity"""
        entity = self.get_entity(entity_id)
        if not entity:
            return None

        return {
            "id": entity.object_id,
            "pos": vec3.to_list(entity.position),
            "velocity": vec3.to_list(entity.velocity),
            "yaw": entity.yaw,
            "pitch": entity.pitch,
            "health": entity.health,
            "on_ground": entity.on_ground,
            "invulnerable_ticks": entity.invulnerablility_ticks,
            "is_sprinting": entity.is_sprinting,
        }
