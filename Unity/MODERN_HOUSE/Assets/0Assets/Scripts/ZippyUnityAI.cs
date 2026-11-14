using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Zippy para Unity 3D. Movimento diferencial, LiDAR 360, odometria,
/// A*, desvio reativo, patrulha, exploração por fronteiras e cobertura.
/// O cenário deve possuir Colliders e estar nas camadas definidas em obstacleMask.
/// </summary>
[RequireComponent(typeof(Rigidbody))]
public class ZippyUnityAI : MonoBehaviour
{
    public enum RobotMode { Manual, Idle, GoToGoal, Patrol, Explore, Cleaning }
    public enum RobotState { Stopped, Manual, Planning, GoingToGoal, AvoidingObstacle, GoalReached, NoRoute }

    [Header("Referências e camadas")]
    public Transform lidarOrigin;
    public LayerMask obstacleMask;
    public LayerMask groundMask;
    public Camera sceneCamera;

    [Header("Movimento diferencial")]
    [Min(0.01f)] public float wheelBase = 0.28f;
    [Min(0.01f)] public float maxWheelSpeed = 1.0f;
    [Min(0.01f)] public float maxLinearSpeed = 0.75f;
    [Min(0.01f)] public float maxAngularSpeed = 2.4f;
    [Min(0.01f)] public float wheelAcceleration = 2.2f;
    [Min(0.01f)] public float robotRadius = 0.22f;
    public float groundHeightOffset = 0.02f;

    [Header("LiDAR")]
    [Range(12, 720)] public int lidarRays = 360;
    [Min(0.2f)] public float lidarRange = 5f;
    [Min(0.01f)] public float lidarMinRange = 0.10f;
    [Min(0.01f)] public float lidarUpdateInterval = 0.10f;
    public bool lidarNoise = true;
    [Range(0f, 0.1f)] public float lidarNoiseMeters = 0.01f;
    [Range(0f, 0.2f)] public float lidarDropRate = 0.015f;

    [Header("Segurança e controle")]
    public float criticalDistance = 0.40f;
    public float obstacleReleaseDistance = 0.60f;
    public float waypointTolerance = 0.18f;
    public float finalGoalTolerance = 0.22f;
    public float lookAhead = 0.25f;
    public float kpAngle = 2.8f;
    public float kpDistance = 1.1f;
    public float freeSpeed = 0.50f;

    [Header("Grade para A* e mapeamento")]
    public Vector2 mapSize = new Vector2(20f, 20f);
    public Vector3 mapCenter = Vector3.zero;
    [Range(0.10f, 1f)] public float cellSize = 0.25f;
    [Min(0f)] public float safetyMargin = 0.08f;
    public float mapFloorY = 0f;
    [Range(0.1f, 5f)] public float mapUpdateInterval = 0.5f;

    [Header("Destinos de patrulha")]
    public Transform[] patrolPoints;
    public bool loopPatrol = true;

    [Header("Teclas")]
    public KeyCode stopKey = KeyCode.S;
    public KeyCode modeKey = KeyCode.Tab;
    public KeyCode patrolKey = KeyCode.P;
    public KeyCode exploreKey = KeyCode.E;
    public KeyCode cleaningKey = KeyCode.Alpha7;

    [Header("Diagnóstico")]
    public bool drawLidar = true;
    public bool drawPath = true;
    public bool showDebugMessages = true;

    public RobotMode Mode { get; private set; } = RobotMode.Idle;
    public RobotState State { get; private set; } = RobotState.Stopped;
    public float[] Scan { get; private set; }
    public float FrontDistance { get; private set; }
    public float LeftDistance { get; private set; }
    public float RightDistance { get; private set; }
    public Vector3 OdometryPosition { get; private set; }
    public float OdometryHeading { get; private set; }
    public float DistanceTravelled { get; private set; }
    public int CollisionCount { get; private set; }
    public float MapCoverage { get; private set; }
    public Vector3? CurrentGoal { get; private set; }

    Rigidbody body;
    float leftWheelTarget, rightWheelTarget, leftWheelSpeed, rightWheelSpeed;
    float lidarTimer, mapTimer, replanTimer, avoidanceDirection = 1f;
    float odomLastLeft, odomLastRight;
    Vector3 lastRealPosition;
    bool wasColliding;
    Vector2 manualCommand;
    readonly List<Vector3> path = new List<Vector3>();
    int pathIndex;
    int patrolIndex;
    int cleaningIndex;
    readonly HashSet<int> visitedCells = new HashSet<int>();
    readonly HashSet<int> cleanedCells = new HashSet<int>();
    float[,] logOdds;
    bool[,] seenCells;
    bool[,] walkable;
    int gridWidth, gridHeight;
    bool gridReady;
    string statusMessage = "Zippy pronto";

    static readonly Vector2Int[] Neighbours = {
        new Vector2Int(1,0), new Vector2Int(-1,0), new Vector2Int(0,1), new Vector2Int(0,-1),
        new Vector2Int(1,1), new Vector2Int(1,-1), new Vector2Int(-1,1), new Vector2Int(-1,-1)
    };

    void Awake()
    {
        body = GetComponent<Rigidbody>();
        body.constraints |= RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
        body.interpolation = RigidbodyInterpolation.Interpolate;
        body.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
        if (sceneCamera == null) sceneCamera = Camera.main;
        if (lidarOrigin == null) lidarOrigin = transform;
        Scan = new float[lidarRays];
        OdometryPosition = transform.position;
        OdometryHeading = transform.eulerAngles.y * Mathf.Deg2Rad;
        lastRealPosition = transform.position;
        BuildGrid();
        TakeLidarScan();
        statusMessage = "LiDAR e grade inicializados";
    }

    void Update()
    {
        ReadKeyboard();
        if (Mode == RobotMode.Manual) ReadManualInput();
        if (Input.GetMouseButtonDown(0)) TrySetGoalFromMouse();

        lidarTimer += Time.deltaTime;
        if (lidarTimer >= lidarUpdateInterval)
        {
            lidarTimer = 0f;
            TakeLidarScan();
            UpdateLearnedMap();
        }

        mapTimer += Time.deltaTime;
        if (mapTimer >= mapUpdateInterval)
        {
            mapTimer = 0f;
            UpdateCoverage();
        }

        if (drawLidar) DrawLidarRays();
        if (drawPath) DrawPath();
    }

    void FixedUpdate()
    {
        UpdateOdometry(Time.fixedDeltaTime);
        if (Mode == RobotMode.Manual)
        {
            SetVelocityCommand(manualCommand.x * maxLinearSpeed, manualCommand.y * maxAngularSpeed);
        }
        else if (Mode == RobotMode.GoToGoal || Mode == RobotMode.Patrol || Mode == RobotMode.Explore || Mode == RobotMode.Cleaning)
        {
            AutonomousStep(Time.fixedDeltaTime);
        }
        else SetVelocityCommand(0f, 0f);

        ApplyDifferentialDrive(Time.fixedDeltaTime);
    }

    void ReadKeyboard()
    {
        if (Input.GetKeyDown(stopKey)) StopRobot();
        if (Input.GetKeyDown(modeKey))
        {
            if (Mode == RobotMode.Manual) { Mode = RobotMode.Idle; StopRobot(); }
            else { Mode = RobotMode.Manual; ClearPath(); State = RobotState.Manual; }
        }
        if (Input.GetKeyDown(patrolKey)) StartPatrol();
        if (Input.GetKeyDown(exploreKey)) StartExploration();
        if (Input.GetKeyDown(cleaningKey)) StartCleaning();
    }

    void ReadManualInput()
    {
        float linear = Input.GetAxisRaw("Vertical");
        float angular = Input.GetAxisRaw("Horizontal");
        manualCommand = new Vector2(linear, angular);
    }

    void TrySetGoalFromMouse()
    {
        if (sceneCamera == null) return;
        Ray ray = sceneCamera.ScreenPointToRay(Input.mousePosition);
        if (Physics.Raycast(ray, out RaycastHit hit, 500f, groundMask, QueryTriggerInteraction.Ignore))
            SetGoal(hit.point, "Destino selecionado");
    }

    public void SetGoal(Vector3 worldPosition, string goalName = "Destino")
    {
        Vector3 goal = new Vector3(worldPosition.x, transform.position.y, worldPosition.z);
        CurrentGoal = goal;
        Mode = RobotMode.GoToGoal;
        PlanPath(goal);
    }

    public void StopRobot()
    {
        Mode = RobotMode.Idle;
        State = RobotState.Stopped;
        CurrentGoal = null;
        ClearPath();
        manualCommand = Vector2.zero;
        SetVelocityCommand(0f, 0f);
        statusMessage = "Robô parado";
    }

    public void StartPatrol()
    {
        if (patrolPoints == null || patrolPoints.Length == 0) { Log("Configure os destinos de patrulha no Inspector."); return; }
        Mode = RobotMode.Patrol;
        patrolIndex = Mathf.Clamp(patrolIndex, 0, patrolPoints.Length - 1);
        GoToNextPatrolPoint();
        Log("Modo patrulha iniciado");
    }

    void GoToNextPatrolPoint()
    {
        if (patrolPoints == null || patrolPoints.Length == 0) { StopRobot(); return; }
        int attempts = 0;
        while (attempts < patrolPoints.Length && patrolPoints[patrolIndex] == null)
        {
            patrolIndex = (patrolIndex + 1) % patrolPoints.Length;
            attempts++;
        }
        if (attempts >= patrolPoints.Length) { StopRobot(); return; }
        CurrentGoal = patrolPoints[patrolIndex].position;
        PlanPath(CurrentGoal.Value);
    }

    public void StartExploration()
    {
        if (!gridReady) BuildGrid();
        Mode = RobotMode.Explore;
        visitedCells.Clear();
        ChooseExplorationFrontier();
        Log("Exploração autônoma iniciada");
    }

    public void StartCleaning()
    {
        if (!gridReady) BuildGrid();
        Mode = RobotMode.Cleaning;
        cleanedCells.Clear();
        cleaningIndex = 0;
        BuildCleaningTargets();
        Log("Cobertura automática iniciada");
    }

    void AutonomousStep(float dt)
    {
        if (CurrentGoal == null || path.Count == 0)
        {
            SetVelocityCommand(0f, 0f);
            if (Mode == RobotMode.Explore) ChooseExplorationFrontier();
            else if (Mode == RobotMode.Cleaning) GoToNextCleaningTarget();
            else if (Mode == RobotMode.Patrol) GoToNextPatrolPoint();
            return;
        }

        if (FrontDistance < criticalDistance)
        {
            if (State != RobotState.AvoidingObstacle)
            {
                State = RobotState.AvoidingObstacle;
                avoidanceDirection = LeftDistance >= RightDistance ? -1f : 1f;
                Log("Obstáculo detectado: freando e girando");
            }
            SetVelocityCommand(0f, avoidanceDirection * Mathf.Min(1.0f, maxAngularSpeed));
            if (FrontDistance > obstacleReleaseDistance)
            {
                State = RobotState.Planning;
                PlanPath(CurrentGoal.Value);
            }
            return;
        }

        if (State == RobotState.AvoidingObstacle) State = RobotState.GoingToGoal;
        Vector3 finalGoal = CurrentGoal.Value;
        if (Vector3.Distance(Flat(transform.position), Flat(finalGoal)) <= finalGoalTolerance)
        {
            OnGoalReached();
            return;
        }

        while (pathIndex < path.Count - 1 && Vector3.Distance(Flat(transform.position), Flat(path[pathIndex])) < waypointTolerance)
            pathIndex++;

        Vector3 target = GetLookAheadTarget();
        Vector3 delta = Flat(target - transform.position);
        if (delta.sqrMagnitude < 0.0001f) { SetVelocityCommand(0f, 0f); return; }
        float desiredHeading = Mathf.Atan2(delta.x, delta.z);
        float currentHeading = transform.eulerAngles.y * Mathf.Deg2Rad;
        float angleError = Mathf.DeltaAngle(currentHeading * Mathf.Rad2Deg, desiredHeading * Mathf.Rad2Deg) * Mathf.Deg2Rad;
        float omega = Mathf.Clamp(kpAngle * angleError, -maxAngularSpeed, maxAngularSpeed);
        float distance = Vector3.Distance(Flat(transform.position), Flat(path[pathIndex]));
        float alignment = Mathf.Pow(Mathf.Max(0f, Mathf.Cos(angleError)), 4f);
        float speed = Mathf.Clamp(kpDistance * distance, 0.06f, maxLinearSpeed * 0.55f) * alignment;
        speed *= Mathf.Clamp01((FrontDistance - criticalDistance * 0.5f) / Mathf.Max(0.1f, obstacleReleaseDistance));
        SetVelocityCommand(speed, omega);
        State = RobotState.GoingToGoal;

        if (Mode == RobotMode.Explore) MarkVisitedCurrentCell();
        if (Mode == RobotMode.Cleaning) MarkCleanedCurrentCell();

        replanTimer += dt;
        if (replanTimer > 1.0f)
        {
            replanTimer = 0f;
            if (!PathStillClear()) PlanPath(finalGoal);
        }
    }

    Vector3 GetLookAheadTarget()
    {
        if (pathIndex <= 0 || pathIndex >= path.Count) return path[Mathf.Clamp(pathIndex, 0, path.Count - 1)];
        Vector3 a = path[pathIndex - 1];
        Vector3 b = path[pathIndex];
        Vector3 segment = Flat(b - a);
        float length = segment.magnitude;
        if (length < 0.001f) return b;
        Vector3 direction = segment / length;
        float projection = Mathf.Clamp(Vector3.Dot(Flat(transform.position - a), direction), 0f, length);
        return a + direction * Mathf.Min(length, projection + lookAhead);
    }

    void OnGoalReached()
    {
        SetVelocityCommand(0f, 0f);
        State = RobotState.GoalReached;
        ClearPath();
        if (Mode == RobotMode.Patrol)
        {
            patrolIndex++;
            if (patrolIndex >= patrolPoints.Length)
            {
                if (loopPatrol) patrolIndex = 0;
                else { StopRobot(); return; }
            }
            GoToNextPatrolPoint();
        }
        else if (Mode == RobotMode.Explore) ChooseExplorationFrontier();
        else if (Mode == RobotMode.Cleaning) GoToNextCleaningTarget();
        else { Mode = RobotMode.Idle; CurrentGoal = null; Log("Destino alcançado"); }
    }

    void SetVelocityCommand(float linear, float angular)
    {
        linear = Mathf.Clamp(linear, -maxLinearSpeed, maxLinearSpeed);
        angular = Mathf.Clamp(angular, -maxAngularSpeed, maxAngularSpeed);
        float left = linear - angular * wheelBase * 0.5f;
        float right = linear + angular * wheelBase * 0.5f;
        float max = Mathf.Max(Mathf.Abs(left), Mathf.Abs(right));
        if (max > maxWheelSpeed)
        {
            float scale = maxWheelSpeed / max;
            left *= scale;
            right *= scale;
        }
        leftWheelTarget = left;
        rightWheelTarget = right;
    }

    void ApplyDifferentialDrive(float dt)
    {
        leftWheelSpeed = Mathf.MoveTowards(leftWheelSpeed, leftWheelTarget, wheelAcceleration * dt);
        rightWheelSpeed = Mathf.MoveTowards(rightWheelSpeed, rightWheelTarget, wheelAcceleration * dt);
        float linear = (leftWheelSpeed + rightWheelSpeed) * 0.5f;
        float angular = (rightWheelSpeed - leftWheelSpeed) / Mathf.Max(0.01f, wheelBase);
        float deltaHeading = angular * dt;
        Quaternion rotation = Quaternion.Euler(0f, deltaHeading * Mathf.Rad2Deg, 0f) * body.rotation;
        Vector3 forward = body.rotation * Vector3.forward;
        Vector3 displacement = forward * (linear * dt);
        displacement.y = 0f;
        body.MovePosition(body.position + displacement);
        body.MoveRotation(rotation);
        if (body.linearVelocity.magnitude > maxLinearSpeed * 1.5f) body.linearVelocity = Vector3.ClampMagnitude(body.linearVelocity, maxLinearSpeed * 1.5f);
    }

    void UpdateOdometry(float dt)
    {
        float linear = (leftWheelSpeed + rightWheelSpeed) * 0.5f;
        float angular = (rightWheelSpeed - leftWheelSpeed) / Mathf.Max(0.01f, wheelBase);
        OdometryHeading = Mathf.Repeat(OdometryHeading + angular * dt + Mathf.PI, 2f * Mathf.PI) - Mathf.PI;
        OdometryPosition += new Vector3(Mathf.Sin(OdometryHeading), 0f, Mathf.Cos(OdometryHeading)) * linear * dt;
        DistanceTravelled += Vector3.Distance(Flat(transform.position), Flat(lastRealPosition));
        lastRealPosition = transform.position;
    }

    void TakeLidarScan()
    {
        if (Scan == null || Scan.Length != lidarRays) Scan = new float[lidarRays];
        Vector3 origin = lidarOrigin != null ? lidarOrigin.position : transform.position;
        Quaternion baseRotation = lidarOrigin != null ? lidarOrigin.rotation : transform.rotation;
        for (int i = 0; i < lidarRays; i++)
        {
            float angle = (float)i / lidarRays * 360f;
            Vector3 direction = Quaternion.AngleAxis(angle, Vector3.up) * baseRotation * Vector3.forward;
            float distance = lidarRange;
            if (Physics.Raycast(origin, direction, out RaycastHit hit, lidarRange, obstacleMask, QueryTriggerInteraction.Ignore))
                distance = hit.distance;
            if (lidarNoise)
            {
                if (UnityEngine.Random.value < lidarDropRate) distance = 0f;
                else if (distance < lidarRange) distance = Mathf.Max(lidarMinRange, distance + UnityEngine.Random.Range(-lidarNoiseMeters, lidarNoiseMeters));
            }
            Scan[i] = distance;
        }
        ProcessScanSectors();
    }

    void ProcessScanSectors()
    {
        FrontDistance = SectorMinimum(345f, 15f);
        LeftDistance = SectorMinimum(225f, 315f);
        RightDistance = SectorMinimum(45f, 135f);
    }

    float SectorMinimum(float startAngle, float endAngle)
    {
        float min = lidarRange;
        for (int i = 0; i < Scan.Length; i++)
        {
            float a = (float)i / Scan.Length * 360f;
            bool inside = startAngle <= endAngle ? a >= startAngle && a <= endAngle : a >= startAngle || a <= endAngle;
            if (inside && Scan[i] >= lidarMinRange && Scan[i] < min) min = Scan[i];
        }
        return min;
    }

    void BuildGrid()
    {
        gridWidth = Mathf.Max(2, Mathf.CeilToInt(mapSize.x / cellSize));
        gridHeight = Mathf.Max(2, Mathf.CeilToInt(mapSize.y / cellSize));
        walkable = new bool[gridWidth, gridHeight];
        logOdds = new float[gridWidth, gridHeight];
        seenCells = new bool[gridWidth, gridHeight];
        float checkRadius = robotRadius + safetyMargin;
        for (int x = 0; x < gridWidth; x++)
        for (int z = 0; z < gridHeight; z++)
        {
            Vector3 p = CellToWorld(x, z);
            walkable[x, z] = !Physics.CheckSphere(p, checkRadius, obstacleMask, QueryTriggerInteraction.Ignore);
        }
        gridReady = true;
    }

    Vector3 CellToWorld(int x, int z)
    {
        return new Vector3(mapCenter.x - mapSize.x * 0.5f + (x + 0.5f) * cellSize,
                           mapFloorY,
                           mapCenter.z - mapSize.y * 0.5f + (z + 0.5f) * cellSize);
    }

    Vector2Int WorldToCell(Vector3 p)
    {
        int x = Mathf.FloorToInt((p.x - (mapCenter.x - mapSize.x * 0.5f)) / cellSize);
        int z = Mathf.FloorToInt((p.z - (mapCenter.z - mapSize.y * 0.5f)) / cellSize);
        return new Vector2Int(x, z);
    }

    bool InGrid(int x, int z) { return x >= 0 && z >= 0 && x < gridWidth && z < gridHeight; }

    void UpdateLearnedMap()
    {
        if (!gridReady || Scan == null) return;
        Vector3 origin = lidarOrigin != null ? lidarOrigin.position : transform.position;
        for (int i = 0; i < Scan.Length; i++)
        {
            float distance = Scan[i];
            if (distance < lidarMinRange) continue;
            float angle = (float)i / Scan.Length * 360f;
            Vector3 direction = Quaternion.AngleAxis(angle, Vector3.up) * (lidarOrigin != null ? lidarOrigin.forward : transform.forward);
            float limit = Mathf.Min(distance, lidarRange);
            for (float d = 0.2f; d < limit - 0.05f; d += cellSize * 0.75f)
            {
                Vector2Int c = WorldToCell(origin + direction * d);
                if (InGrid(c.x, c.y)) { logOdds[c.x, c.y] = Mathf.Max(-3f, logOdds[c.x, c.y] - 0.35f); seenCells[c.x, c.y] = true; }
            }
            if (distance < lidarRange - 0.02f)
            {
                Vector2Int hit = WorldToCell(origin + direction * distance);
                if (InGrid(hit.x, hit.y)) { logOdds[hit.x, hit.y] = Mathf.Min(3f, logOdds[hit.x, hit.y] + 0.8f); seenCells[hit.x, hit.y] = true; }
            }
        }
    }

    void UpdateCoverage()
    {
        if (!gridReady) return;
        int free = 0, known = 0;
        for (int x = 0; x < gridWidth; x++)
        for (int z = 0; z < gridHeight; z++)
        {
            if (!walkable[x, z]) continue;
            free++;
            if (seenCells[x, z]) known++;
        }
        MapCoverage = free > 0 ? (float)known / free : 0f;
    }

    public bool PlanPath(Vector3 destination)
    {
        if (!gridReady) BuildGrid();
        State = RobotState.Planning;
        path.Clear();
        Vector2Int start = FindNearestFree(WorldToCell(transform.position));
        Vector2Int goal = FindNearestFree(WorldToCell(destination));
        if (start.x < 0 || goal.x < 0)
        {
            State = RobotState.NoRoute;
            Log("A*: não encontrei uma célula livre para origem/destino");
            return false;
        }
        List<Vector2Int> cells = AStar(start, goal);
        if (cells == null || cells.Count == 0)
        {
            State = RobotState.NoRoute;
            Log("A*: não existe rota até o destino");
            return false;
        }
        List<Vector2Int> smooth = SmoothPath(cells);
        foreach (Vector2Int c in smooth) path.Add(CellToWorld(c.x, c.y));
        if (path.Count == 0) path.Add(destination);
        pathIndex = path.Count > 1 ? 1 : 0;
        State = RobotState.GoingToGoal;
        replanTimer = 0f;
        return true;
    }

    Vector2Int FindNearestFree(Vector2Int origin)
    {
        if (InGrid(origin.x, origin.y) && walkable[origin.x, origin.y]) return origin;
        for (int r = 1; r < Mathf.Max(gridWidth, gridHeight); r++)
        {
            for (int x = origin.x - r; x <= origin.x + r; x++)
            for (int z = origin.y - r; z <= origin.y + r; z++)
            {
                if (!InGrid(x, z) || Mathf.Abs(x - origin.x) != r && Mathf.Abs(z - origin.y) != r) continue;
                if (walkable[x, z]) return new Vector2Int(x, z);
            }
        }
        return new Vector2Int(-1, -1);
    }

    List<Vector2Int> AStar(Vector2Int start, Vector2Int goal)
    {
        int total = gridWidth * gridHeight;
        float[] g = new float[total];
        int[] parent = new int[total];
        bool[] closed = new bool[total];
        for (int i = 0; i < total; i++) { g[i] = float.PositiveInfinity; parent[i] = -1; }
        int startId = start.y * gridWidth + start.x;
        int goalId = goal.y * gridWidth + goal.x;
        List<int> open = new List<int> { startId };
        g[startId] = 0f;
        while (open.Count > 0)
        {
            int bestIndex = 0;
            float bestF = float.PositiveInfinity;
            for (int i = 0; i < open.Count; i++)
            {
                int id = open[i];
                int x = id % gridWidth, z = id / gridWidth;
                float f = g[id] + Heuristic(x, z, goal.x, goal.y);
                if (f < bestF) { bestF = f; bestIndex = i; }
            }
            int current = open[bestIndex];
            open.RemoveAt(bestIndex);
            if (current == goalId) return ReconstructPath(parent, current);
            if (closed[current]) continue;
            closed[current] = true;
            int cx = current % gridWidth, cz = current / gridWidth;
            foreach (Vector2Int n in Neighbours)
            {
                int nx = cx + n.x, nz = cz + n.y;
                if (!InGrid(nx, nz) || !walkable[nx, nz]) continue;
                if (n.x != 0 && n.y != 0 && (!walkable[cx + n.x, cz] || !walkable[cx, cz + n.y])) continue;
                int ni = nz * gridWidth + nx;
                if (closed[ni]) continue;
                float step = n.x != 0 && n.y != 0 ? 1.4142f : 1f;
                float nearPenalty = HasBlockedNeighbour(nx, nz) ? 0.20f : 0f;
                float tentative = g[current] + step + nearPenalty;
                if (tentative < g[ni])
                {
                    g[ni] = tentative;
                    parent[ni] = current;
                    if (!open.Contains(ni)) open.Add(ni);
                }
            }
        }
        return null;
    }

    float Heuristic(int x, int z, int gx, int gz)
    {
        int dx = Mathf.Abs(x - gx), dz = Mathf.Abs(z - gz);
        return Mathf.Max(dx, dz) + 0.4142f * Mathf.Min(dx, dz);
    }

    bool HasBlockedNeighbour(int x, int z)
    {
        for (int dx = -1; dx <= 1; dx++)
        for (int dz = -1; dz <= 1; dz++)
            if (InGrid(x + dx, z + dz) && !walkable[x + dx, z + dz]) return true;
        return false;
    }

    List<Vector2Int> ReconstructPath(int[] parent, int current)
    {
        List<Vector2Int> result = new List<Vector2Int>();
        while (current >= 0)
        {
            result.Add(new Vector2Int(current % gridWidth, current / gridWidth));
            current = parent[current];
        }
        result.Reverse();
        return result;
    }

    List<Vector2Int> SmoothPath(List<Vector2Int> cells)
    {
        if (cells.Count <= 2) return cells;
        List<Vector2Int> result = new List<Vector2Int> { cells[0] };
        int i = 0;
        while (i < cells.Count - 1)
        {
            int j = cells.Count - 1;
            while (j > i + 1 && !GridLineClear(cells[i], cells[j])) j--;
            result.Add(cells[j]);
            i = j;
        }
        return result;
    }

    bool GridLineClear(Vector2Int a, Vector2Int b)
    {
        Vector3 start = CellToWorld(a.x, a.y) + Vector3.up * 0.1f;
        Vector3 end = CellToWorld(b.x, b.y) + Vector3.up * 0.1f;
        Vector3 delta = end - start;
        if (Physics.SphereCast(start, robotRadius + safetyMargin, delta.normalized, out _, delta.magnitude, obstacleMask, QueryTriggerInteraction.Ignore)) return false;
        int steps = Mathf.CeilToInt(delta.magnitude / (cellSize * 0.25f));
        for (int i = 0; i <= steps; i++)
        {
            float t = steps > 0 ? (float)i / steps : 0f;
            Vector2Int c = WorldToCell(Vector3.Lerp(start, end, t));
            if (!InGrid(c.x, c.y) || !walkable[c.x, c.y]) return false;
        }
        return true;
    }

    bool PathStillClear()
    {
        if (path.Count < 2) return true;
        for (int i = Mathf.Max(0, pathIndex - 1); i < path.Count - 1; i++)
        {
            Vector3 a = path[i] + Vector3.up * 0.1f;
            Vector3 b = path[i + 1] + Vector3.up * 0.1f;
            Vector3 d = b - a;
            if (Physics.SphereCast(a, robotRadius + safetyMargin, d.normalized, out _, d.magnitude, obstacleMask, QueryTriggerInteraction.Ignore)) return false;
        }
        return true;
    }

    void ChooseExplorationFrontier()
    {
        if (!gridReady) return;
        Vector2Int current = WorldToCell(transform.position);
        Vector2Int best = new Vector2Int(-1, -1);
        float bestScore = float.PositiveInfinity;
        for (int x = 1; x < gridWidth - 1; x++)
        for (int z = 1; z < gridHeight - 1; z++)
        {
            if (!walkable[x, z] || !seenCells[x, z] || logOdds[x, z] > 0.8f) continue;
            int id = z * gridWidth + x;
            if (visitedCells.Contains(id)) continue;
            bool frontier = false;
            for (int dx = -1; dx <= 1; dx++)
            for (int dz = -1; dz <= 1; dz++)
                if (!seenCells[x + dx, z + dz]) frontier = true;
            if (!frontier) continue;
            float score = Vector2.Distance(new Vector2(x, z), new Vector2(current.x, current.y));
            if (score < bestScore) { bestScore = score; best = new Vector2Int(x, z); }
        }
        if (best.x < 0)
        {
            // Se ainda não há fronteira conhecida, escolhe uma célula livre não visitada próxima.
            for (int x = 0; x < gridWidth; x++)
            for (int z = 0; z < gridHeight; z++)
            {
                int id = z * gridWidth + x;
                if (!walkable[x, z] || visitedCells.Contains(id)) continue;
                float score = Vector2.Distance(new Vector2(x, z), new Vector2(current.x, current.y));
                if (score < bestScore) { bestScore = score; best = new Vector2Int(x, z); }
            }
        }
        if (best.x < 0) { StopRobot(); Log("Exploração concluída: não há novas células-alvo"); return; }
        visitedCells.Add(best.y * gridWidth + best.x);
        CurrentGoal = CellToWorld(best.x, best.y);
        PlanPath(CurrentGoal.Value);
    }

    void MarkVisitedCurrentCell()
    {
        Vector2Int c = WorldToCell(transform.position);
        if (InGrid(c.x, c.y)) visitedCells.Add(c.y * gridWidth + c.x);
    }

    void BuildCleaningTargets()
    {
        cleaningTargets.Clear();
        int stride = Mathf.Max(1, Mathf.RoundToInt(0.75f / cellSize));
        for (int z = 0; z < gridHeight; z += stride)
        {
            bool reverse = (z / stride) % 2 == 1;
            for (int k = 0; k < gridWidth; k += stride)
            {
                int x = reverse ? gridWidth - 1 - k : k;
                if (InGrid(x, z) && walkable[x, z]) cleaningTargets.Add(CellToWorld(x, z));
            }
        }
        cleaningIndex = 0;
        GoToNextCleaningTarget();
    }

    readonly List<Vector3> cleaningTargets = new List<Vector3>();

    void GoToNextCleaningTarget()
    {
        while (cleaningIndex < cleaningTargets.Count)
        {
            Vector3 target = cleaningTargets[cleaningIndex++];
            if (PlanPath(target)) { CurrentGoal = target; return; }
        }
        Mode = RobotMode.Idle;
        CurrentGoal = null;
        State = RobotState.GoalReached;
        Log("Cobertura concluída");
    }

    void MarkCleanedCurrentCell()
    {
        Vector2Int c = WorldToCell(transform.position);
        if (InGrid(c.x, c.y)) cleanedCells.Add(c.y * gridWidth + c.x);
    }

    void ClearPath() { path.Clear(); pathIndex = 0; }

    void DrawLidarRays()
    {
        Vector3 origin = lidarOrigin != null ? lidarOrigin.position : transform.position;
        for (int i = 0; i < Scan.Length; i++)
        {
            float d = Scan[i];
            if (d < lidarMinRange) continue;
            float angle = (float)i / Scan.Length * 360f;
            Vector3 direction = Quaternion.AngleAxis(angle, Vector3.up) * (lidarOrigin != null ? lidarOrigin.forward : transform.forward);
            Color color = d < criticalDistance ? Color.red : Color.green;
            Debug.DrawLine(origin, origin + direction * Mathf.Min(d, lidarRange), color, lidarUpdateInterval);
        }
    }

    void DrawPath()
    {
        for (int i = 0; i < path.Count - 1; i++) Debug.DrawLine(path[i] + Vector3.up * 0.08f, path[i + 1] + Vector3.up * 0.08f, Color.cyan);
        if (CurrentGoal.HasValue) Debug.DrawLine(transform.position, CurrentGoal.Value, Color.yellow);
    }

    void OnCollisionEnter(Collision collision)
    {
        if (!wasColliding) CollisionCount++;
        wasColliding = true;
        if (Mode != RobotMode.Manual && CurrentGoal.HasValue)
        {
            State = RobotState.Planning;
            PlanPath(CurrentGoal.Value);
        }
    }

    void OnCollisionExit(Collision collision) { wasColliding = false; }

    void Log(string message)
    {
        statusMessage = message;
        if (showDebugMessages) Debug.Log("[Zippy] " + message, this);
    }

    static Vector3 Flat(Vector3 v) { return new Vector3(v.x, 0f, v.z); }

    void OnGUI()
    {
        GUI.Box(new Rect(10, 10, 250, 148), "ZIPPY  |  Robô móvel");
        GUI.Label(new Rect(22, 38, 230, 20), "Modo: " + Mode);
        GUI.Label(new Rect(22, 58, 230, 20), "Estado: " + State);
        GUI.Label(new Rect(22, 78, 230, 20), string.Format("LiDAR F/E/D: {0:F2} / {1:F2} / {2:F2} m", FrontDistance, LeftDistance, RightDistance));
        GUI.Label(new Rect(22, 98, 230, 20), string.Format("Odometria: {0:F1}, {1:F1}", OdometryPosition.x, OdometryPosition.z));
        GUI.Label(new Rect(22, 118, 230, 20), string.Format("Percurso: {0:F1} m | Colisões: {1}", DistanceTravelled, CollisionCount));
        GUI.Label(new Rect(22, 138, 230, 20), string.Format("Mapa conhecido: {0:P0}", MapCoverage));
    }
}
