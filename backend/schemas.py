from pydantic import BaseModel, Field


class MachineInput(BaseModel):
    machine_type: str = Field(
        ...,
        description="Machine type: L, M, or H"
    )

    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    # temperature_difference: float
    # power_proxy: float
