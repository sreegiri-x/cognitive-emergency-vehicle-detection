# Cognitive EV Detection — Problem Statement & Mathematical Formulation

## 1. Problem Statement

Emergency vehicles such as ambulances, fire trucks, and police vehicles need to move through normal traffic quickly. Their movement can cause surrounding vehicles to suddenly change speed, change lanes, brake, or move away from their path.

The problem we are trying to solve is:

> **Given a sequence of vehicle movements and traffic-video observations, identify an emergency vehicle, track its interaction with surrounding vehicles, and predict whether the current traffic situation is moving toward a high-risk state.**

The important part is that we are **not trying to predict an accident with certainty**.

Instead, we want to detect the **early behavioural patterns that indicate increasing risk**.

```text
Normal Traffic
      ↓
Emergency Vehicle Approaches
      ↓
Vehicles React
      ↓
Distance / Speed Changes
      ↓
TTC decreases
      ↓
Risk increases
      ↓
Potential Critical Event
```

The model therefore learns:

$$
Past\ Traffic\ Behaviour \rightarrow Future\ Risk
$$

---

# 2. Emergency Vehicle Identification

The system first needs to identify which vehicle is the emergency vehicle.

Depending on the available I-24 data, this can come from:

1. Dataset-provided vehicle classification, if available.
2. Video-based vehicle detection/classification.
3. Association between video tracks and trajectory tracks.

Once identified, the emergency vehicle becomes the **reference vehicle**.

Let:

$$
E = \text{Emergency Vehicle}
$$

and surrounding vehicles be:

$$
V_1,V_2,...,V_n
$$

We then study:

$$
E \leftrightarrow V_i
$$

for every relevant nearby vehicle.

---

# 3. Vehicle State

At time \(t\), each vehicle is represented by a state vector:

$$
X_i(t) =
[x_i,y_i,v_{x,i},v_{y,i},a_{x,i},a_{y,i}]
$$

where:

* \(x,y\) = vehicle position
* \(v_x,v_y\) = velocity
* \(a_x,a_y\) = acceleration

Instead of looking at one instant, we observe a sequence:

$$
X_i(t-T),...,X_i(t-1),X_i(t)
$$

This allows the model to understand **how the vehicle is moving**, rather than only where it is.

---

# 4. Distance

For two vehicles \(i\) and \(j\), their Euclidean distance is:

$$
d_{ij}
=
\sqrt{(x_i-x_j)^2+(y_i-y_j)^2}
$$

### Meaning

This tells us how far apart two vehicles are.

For example:

```text
Vehicle A ●────────────● Vehicle B
              d
```

A decreasing distance can indicate that two vehicles are approaching each other.

Distance alone does **not** indicate danger. A large closing speed can make a moderate distance more significant.

---

# 5. Relative Velocity

The relative velocity between two vehicles is:

$$
\Delta v_{ij}=v_i-v_j
$$

For two vehicles travelling in the same direction:

* \(\Delta v > 0\): vehicle \(i\) is moving faster.
* \(\Delta v < 0\): vehicle \(i\) is moving slower.

The important information is whether the distance between vehicles is decreasing.

---

# 6. Time-to-Collision (TTC)

One of the most important features is **Time-to-Collision**.

A simplified longitudinal formulation is:

$$
TTC =
\frac{d}{-\Delta v}
$$

when:

$$
\Delta v < 0
$$

and the vehicles are approaching each other.

### Example

Suppose:

$$
d=30m
$$

and the closing speed is:

$$
\Delta v=-10m/s
$$

Then:

$$
TTC=\frac{30}{10}=3s
$$

This means that, **if the current relative motion continued unchanged**, the estimated collision time would be approximately 3 seconds.

### Why TTC matters

Consider:

```text
Situation A:
Distance = 10 m
Closing speed = 1 m/s

Situation B:
Distance = 30 m
Closing speed = 15 m/s
```

Situation B can become critical much faster despite having a larger distance.

Therefore:

$$
Risk \neq f(distance)
$$

Instead:

$$
Risk = f(distance,\ relative\ velocity,\ acceleration,\ context,...)
$$

TTC is therefore a useful temporal representation of the interaction.

---

# 7. Time Headway (THW)

Time Headway estimates how much time separates a following vehicle from the vehicle ahead.

$$
THW=\frac{d}{v}
$$

where:

* \(d\) = distance to the vehicle ahead
* \(v\) = speed of the following vehicle

Example:

$$
d=40m,\quad v=20m/s
$$

Therefore:

$$
THW=\frac{40}{20}=2s
$$

A decreasing THW can indicate that vehicles are getting closer in time as well as physical distance.

---

# 8. Relative Acceleration

We also consider differences in acceleration:

$$
\Delta a_{ij}=a_i-a_j
$$

This helps identify rapidly changing interactions.

For example:

```text
Emergency Vehicle
       ↓
Accelerating

Nearby Vehicle
       ↓
Braking

        ↓

Relative motion changes rapidly
```

A sudden acceleration or deceleration can therefore provide information about an evolving interaction.

---

# 9. Lane Relationship

Distance alone is insufficient.

Two vehicles can be very close but travelling safely in adjacent lanes.

Therefore, we also consider lane relationships:

$$
L_{ij}
$$

Examples include:

* Same lane
* Adjacent lane
* Lane change
* Cut-in
* Cut-out

This provides **context** to the distance and velocity measurements.

---

# 10. Vehicle Interaction

For an emergency vehicle \(E\) and surrounding vehicle \(i\), we can construct an interaction feature vector:

$$
I_{E,i}(t)
=
[d_{E,i},
\Delta v_{E,i},
\Delta a_{E,i},
TTC_{E,i},
THW_{E,i},
L_{E,i}]
$$

This represents the relationship between the emergency vehicle and another vehicle at time \(t\).

For multiple vehicles:

$$
I_E(t)=
\{I_{E,1}(t),I_{E,2}(t),...,I_{E,n}(t)\}
$$

This is the core of our **traffic interaction representation**.

---

# 11. Why We Use a Temporal Window

A single frame cannot tell us whether a situation is becoming more dangerous.

Instead, we observe:

$$
I(t-T),...,I(t)
$$

For example:

```text
t0   Normal
 ↓
t1   EV approaches
 ↓
t2   Distance decreases
 ↓
t3   Vehicle changes lane
 ↓
t4   Closing speed increases
 ↓
t5   TTC decreases
 ↓
t6   High-risk state
```

The model learns patterns across this sequence.

Therefore:

$$
Risk_{future}
=
f(I_{t-T:t})
$$

---

# 12. Risk Score

The final model produces a continuous risk score:

$$
R_t=f(X_{t-T:t})
$$

where:

$$
0\leq R_t\leq1
$$

A higher value represents a stronger model-estimated risk state.

For example:

```text
0.0 ──────────────── 1.0
Low                  High
```

The value should be interpreted as a **model score** unless we perform probability calibration.

---

# 13. Future Risk Prediction

The most important part of the project is predicting a future state rather than simply classifying the current frame.

We can formulate it as:

$$
R_{t+k}
=
f(X_{t-T:t})
$$

where:

* \(t\) = current time
* \(T\) = observation window
* \(k\) = prediction horizon

For example:

```text
Observe:
t-5 ── t-4 ── t-3 ── t-2 ── t-1 ── t

                         ↓

                    Prediction

                         ↓

                         t+k
```

The model therefore asks:

> **Based on what has happened recently, how risky is the traffic state likely to become in the near future?**

---

# 14. Video Representation

The trajectory information tells us **what vehicles are doing**.

Video provides information about **what is visually happening in the scene**.

A video sequence is passed through VideoMAE:

$$
V_{t-T:t}
\rightarrow
VideoMAE
\rightarrow
F_{video}
$$

where \(F_{video}\) is the learned visual representation.

---

# 15. Trajectory Representation

Similarly, trajectory information is transformed into a learned representation:

$$
X_{t-T:t}
\rightarrow
Trajectory\ Model
\rightarrow
F_{traj}
$$

where \(F_{traj}\) represents the learned behavioural information.

---

# 16. Multimodal Fusion

The two representations are combined:

$$
F_{fusion}
=
[F_{video}\Vert F_{traj}]
$$

where \(\Vert\) means concatenation.

Therefore:

```text
Video ───────► VideoMAE ───────► F_video
                                      │
                                      ▼
                                   Fusion
                                      │
                                      ▼
Trajectory ───► Model ───────────► F_traj
                                      │
                                      ▼
                                Risk Predictor
```

The objective is to combine **visual evidence** and **vehicle-behaviour evidence**.

---

# 17. Explainability

After generating a prediction, we need to determine **why the model produced it**.

The explanation contains two types of evidence.

### Visual evidence

From attention/attribution:

$$
A_{visual}=g(V)
$$

This identifies important frames or spatial regions.

### Behavioural evidence

From trajectory features:

$$
A_{traj}=g(X)
$$

This identifies important factors such as:

* Low TTC
* Decreasing distance
* High relative velocity
* Sudden acceleration
* Lane movement

The two can then be combined:

$$
A_{total}
=
A_{visual}
+
A_{traj}
$$

The exact fusion method will be determined during implementation and experiments.

---

# 18. LLM Explanation

The LLM is **not responsible for predicting risk**.

Instead:

```text
Traffic Data
     ↓
ML Model
     ↓
Risk Score
     ↓
Evidence Extraction
     ↓
Structured Evidence
     ↓
LLM
     ↓
Human-readable explanation
```

For example:

```text
Risk Score: 0.87

TTC: 1.2 s
Relative velocity: 18 m/s
Lane change: detected
Important vehicles: 42, 57
```

The LLM converts these verified model outputs into a readable explanation.

This prevents the language model from independently inventing the reason for a prediction.

---

# 19. Complete Mathematical Flow

The complete process can be summarized as:

$$
X_{t-T:t}
\rightarrow
I_{t-T:t}
\rightarrow
F_{traj}
$$

and:

$$
V_{t-T:t}
\rightarrow
VideoMAE
\rightarrow
F_{video}
$$

then:

$$
F_{fusion}
=
[F_{traj}\Vert F_{video}]
$$

then:

$$
R_{t+k}
=
f(F_{fusion})
$$

and finally:

$$
R_{t+k}
+
Evidence
\rightarrow
Explanation
$$

---

# 20. Core Idea

The fundamental idea behind Cognitive EV Detection is:

$$
\boxed{
Vehicle\ Behaviour
+
Vehicle\ Interaction
+
Temporal\ Evolution
+
Visual\ Context
\rightarrow
Future\ Risk
}
$$

The system therefore does not simply ask:

> **"Is there an emergency vehicle?"**

It asks:

> **"How are surrounding vehicles responding to the emergency vehicle, how is that interaction evolving over time, and does the observed behaviour indicate an increasing level of traffic risk?"**

That is the core problem the project is designed to investigate.
