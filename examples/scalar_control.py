from trivax import ScalarPlant, TrivaxController, run_closed_loop

controller = TrivaxController(initial_action=0.20, step_size=0.04)
plant = ScalarPlant()
records = run_closed_loop(controller, plant, steps=100)

last = records[-1]
print(
    "final",
    {
        "action": round(last["action"], 6),
        "target": round(last["target"], 6),
        "abs_error": round(last["abs_error"], 6),
        "coherence": round(last["coherence"], 6),
    },
)
