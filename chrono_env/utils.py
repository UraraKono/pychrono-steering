import pychrono as chrono
import pychrono.vehicle as veh
import numpy as np
import math

def init_vehicle(self):
    # Create the vehicle system
    my_hmmwv = veh.HMMWV_Full()
    my_hmmwv.SetContactMethod(chrono.ChContactMethod_SMC)
    my_hmmwv.SetChassisFixed(False)
    my_hmmwv.SetInitPosition(chrono.ChCoordsysD(self.ini_pos,chrono.QUNIT))
    my_hmmwv.SetPowertrainType(veh.PowertrainModelType_SHAFTS)
    # my_hmmwv.SetDriveType(veh.DrivelineTypeWV_RWD)
    my_hmmwv.SetDriveType(veh.DrivelineTypeWV_AWD)
    my_hmmwv.SetSteeringType(veh.SteeringTypeWV_PITMAN_ARM)
    my_hmmwv.SetTireType(veh.TireModelType_TMEASY)
    my_hmmwv.SetTireStepSize(self.step_size) # self.step_size
    my_hmmwv.Initialize()

    my_hmmwv.SetChassisVisualizationType(veh.VisualizationType_PRIMITIVES)
    my_hmmwv.SetSuspensionVisualizationType(veh.VisualizationType_PRIMITIVES)
    my_hmmwv.SetSteeringVisualizationType(veh.VisualizationType_PRIMITIVES)
    my_hmmwv.SetWheelVisualizationType(veh.VisualizationType_PRIMITIVES)
    my_hmmwv.SetTireVisualizationType(veh.VisualizationType_PRIMITIVES)

    self.my_hmmwv = my_hmmwv
    return my_hmmwv

def init_terrain(self, friction, reduced_waypoints):
    # Define the patch coordinates
    patch_coords = [[waypoint[1], waypoint[2], 0.0] for waypoint in reduced_waypoints]

    rest_values = [0.01] * len(patch_coords)
    young_modulus_values = [2e7] * len(patch_coords)
    patch_mats = [chrono.ChMaterialSurfaceSMC() for _ in range(len(patch_coords))]
    for i, patch_mat in enumerate(patch_mats):
        patch_mat.SetFriction(friction[i])
        patch_mat.SetRestitution(rest_values[i])
        patch_mat.SetYoungModulus(young_modulus_values[i])

    terrain = veh.RigidTerrain(self.my_hmmwv.GetSystem()) # self.my_hmmwv

    # Base grass terrain
    patch_coords_np = np.array(patch_coords)
    min_x = min(patch_coords_np[:, 0])
    max_x = max(patch_coords_np[:, 0])
    min_y = min(patch_coords_np[:, 1])
    max_y = max(patch_coords_np[:, 1])
    # If the z position is 0, the visualization blinks so much
    base_pos = chrono.ChVectorD((min_x+max_x)/2, (min_y+max_y)/2, -0.05)
    terrainLength = max_x - min_x + 20  # size in X direction
    terrainWidth  = max_y - min_y + 20 # size in Y direction
    patch_mat_base = chrono.ChMaterialSurfaceSMC()
    patch_mat_base.SetFriction(1.4)
    patch_mat_base.SetRestitution(0.01)
    patch_mat_base.SetYoungModulus(2e7)
    base_coord = chrono.ChCoordsysD(base_pos,chrono.QUNIT)
    patch_base = terrain.AddPatch(patch_mat_base, 
                             base_coord, 
                             terrainLength, terrainWidth)
    patch_base.SetTexture(veh.GetDataFile("terrain/textures/grass.jpg"), 200, 200) #concrete, dirt, grass, tile4
    print(veh.GetDataPath())
    # patch_base.SetColor(chrono.ChColor(0.8, 0.8, 0.5))
    

    # Loop over the patch materials and coordinates to add each patch to the terrain
    patches = []
    for i, patch_mat in enumerate(patch_mats):
        coords = patch_coords[i]
        psi = reduced_waypoints[i, 3]
        if i == len(patch_mats) - 1:
            # s = reduced_waypoints[i, 0] - reduced_waypoints[i-1,0]
            s=0
        else:    
            s = reduced_waypoints[i+1, 0] - reduced_waypoints[i,0]

        print("s", s)
        r = chrono.ChQuaternionD()
        r.Q_from_AngZ(psi)
        # print('r',r)
        patch = terrain.AddPatch(patch_mat, chrono.ChCoordsysD(chrono.ChVectorD(coords[0], coords[1], coords[2]), r), 10, s*self.reduced_rate*1.5)
        patches.append(patch)

    # viz_patch = terrain.AddPatch(patch_mats[2], chrono.CSYSNORM, s, s)
    viz_patch = None
    
    # Set color of patch based on friction value
    min_friction = min(friction)
    max_friction = max(friction)
    for i, patch in enumerate(patches):
        # print(friction[i])
        if max_friction == min_friction:
            RGB_value = 1 - friction[i]/1.5
            if RGB_value < 0:
                RGB_value = 0
        else:
            RGB_value = 1 - (friction[i] - min_friction) / (max_friction - min_friction)
        # RGB_value = 1 - (friction[i] - 0.4)
        # print(RGB_value, friction[i])
        patch.SetColor(chrono.ChColor(RGB_value, RGB_value, RGB_value))

    # for patch in patches:
    #     patch.SetTexture(veh.GetDataFile("terrain/textures/concrete.jpg"), 10, 10)

    terrain.Initialize()
    return terrain, viz_patch

def init_irrlicht_vis(ego_vehicle):
    # Create the vehicle Irrlicht interface
    vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
    vis.SetWindowTitle('MPC control')
    vis.SetWindowSize(1280, 1024)
    vis.SetHUDLocation(500, 20)
    vis.Initialize()
    vis.AddLogo()
    vis.AddLightDirectional()
    vis.SetChaseCamera(chrono.ChVectorD(0.0, 0.0, 1.75), 6.0, 4.5)
    vis.AddSkyBox()
    vis.AttachVehicle(ego_vehicle.GetVehicle())

    return vis

def get_vehicle_state(self):
    vehicle = self.my_hmmwv
    pos = vehicle.GetVehicle().GetPos()
    power = vehicle.GetVehicle().GetPowertrain().GetOutputTorque()
    # print("power train torque", power)
    # print("Vehicle position:", pos)
    x = pos.x
    y = pos.y
    rotation = vehicle.GetVehicle().GetRot()
    # print("Vehicle rotation:", rotation)

    # Get the angular velocities of the chassis in the local frame
    chassis_velocity = vehicle.GetVehicle().GetChassis().GetBody().GetWvel_loc()
    yaw_rate = chassis_velocity.z

    euler_angles = rotation.Q_to_Euler123()
    # print("Vehicle rotation (Euler angles):", euler_angles)
    roll_angle = euler_angles.x
    pitch_angle = euler_angles.y
    yaw_angle = euler_angles.z
    print("Vehicle roll angle:", roll_angle)
    print("Vehicle pitch angle:", pitch_angle)
    print("Vehicle yaw angle:", yaw_angle)

    # Get the linear velocity of the chassis in the global frame
    chassis_velocity = vehicle.GetVehicle().GetChassis().GetBody().GetPos_dt()

    # Get the rotation matrix of the chassis
    chassis_rot = vehicle.GetVehicle().GetChassis().GetRot()
    rot_matrix = chrono.ChMatrix33D(chassis_rot)

    # Create an empty ChMatrix33D for the transpose
    transpose_rot_matrix = chrono.ChMatrix33D()

    # Manually set the transpose of the rotation matrix
    transpose_rot_matrix[0, 0] = rot_matrix[0, 0]
    transpose_rot_matrix[1, 0] = rot_matrix[0, 1]
    transpose_rot_matrix[2, 0] = rot_matrix[0, 2]
    transpose_rot_matrix[0, 1] = rot_matrix[1, 0]
    transpose_rot_matrix[1, 1] = rot_matrix[1, 1]
    transpose_rot_matrix[2, 1] = rot_matrix[1, 2]
    transpose_rot_matrix[0, 2] = rot_matrix[2, 0]
    transpose_rot_matrix[1, 2] = rot_matrix[2, 1]
    transpose_rot_matrix[2, 2] = rot_matrix[2, 2]

    # Transform the global frame velocity to the local frame
    chassis_velocity_local = transpose_rot_matrix * chassis_velocity

    # Extract the y-component of the velocity in the local frame
    velocity_y_local = chassis_velocity_local.y

    # Print the velocity in the y direction in the local frame
    # print("Vehicle velocity in y direction (local frame):", velocity_y_local)
    vy = velocity_y_local

    # Extract the y-component of the velocity in the local frame
    velocity_x_local = chassis_velocity_local.x

    # Print the velocity in the x direction in the local frame
    # print("Vehicle velocity in x direction (local frame):", velocity_x_local)
    vx = velocity_x_local

    # Extract the y-component of the velocity
    velocity_y = chassis_velocity.y

    # Print the velocity in the y direction
    # print("Vehicle velocity in y direction:", velocity_y)
    
    # rotation = list(rotation)
    # steering_angle = vehicle.GetVehicle()

    # acc = vehicle.GetVehicle().GetAcc()
    # print("acc", acc)

    max_steering_angle = vehicle.GetVehicle().GetMaxSteeringAngle()
    print("max steering angle:", max_steering_angle)

    # print("steering angle", driver_inputs.m_steering)

    # get vehicle mass
    mass = vehicle.GetVehicle().GetMass()
    # print("Vehicle mass:", mass)

    # Get tire force
    tf_FL = vehicle.GetVehicle().GetTire(0, veh.LEFT).ReportTireForce(self.terrain)
    tf_FR = vehicle.GetVehicle().GetTire(0, veh.RIGHT).ReportTireForce(self.terrain)
    tf_RL = vehicle.GetVehicle().GetTire(1, veh.LEFT).ReportTireForce(self.terrain)
    tf_RR = vehicle.GetVehicle().GetTire(1, veh.RIGHT).ReportTireForce(self.terrain)
    # print("   Front left:  ", tf_FL.force.x, " ", tf_FL.force.y, " ", tf_FL.force.z)
    # print("   Front right: ", tf_FR.force.x, " ", tf_FR.force.y, " ", tf_FR.force.z)
    # print("   Rear left:   ", tf_RL.force.x, " ", tf_RL.force.y, " ", tf_RL.force.z)
    # print("   Rear right:  ", tf_RR.force.x, " ", tf_RR.force.y, " ", tf_RR.force.z)

    # get tractive force
    mu = 0.8

    Fx_left = tf_FL.force.z * mu
    Fx_right = tf_FR.force.z * mu
    Rx_left = tf_RL.force.z * mu
    Rx_right = tf_RR.force.z * mu
    # print("   Fx_left:  ", Fx_left)
    # print("   Fx_right: ", Fx_right)
    # print("   Rx_left:   ", Rx_left)
    # print("   Rx_right:  ", Rx_right)

    # my_vehicle = veh.ChWheeledVehicle("my_vehicle")
    # my_driver = veh.ChDriver(vehicle.GetVehicle()) #This command does NOT work. Never use ChDriver!
    # throttle = my_driver.GetThrottle()
    # steering = my_driver.GetSteering() # steering input [-1,+1]
    # braking = my_driver.GetBraking()

    # steering = veh.DriverInputs().m_steering
    steering = self.driver_inputs.m_steering
    # print("steering ratio sriver inputs", steering)
    # inputs = my_driver.GetInputs()

    driver_glob_location = vehicle.GetVehicle().GetDriverPos()  # global location of the driver
    # (Include the rest of the code inside the original loop)


    # vehicle state for single-track model
    vehicle_state = np.array([x,  # x
                              y,  # y
                              vx,  # vx
                              yaw_angle,  # yaw angle
                              vy,  # vy
                              yaw_rate,  # yaw rate
                              steering*max_steering_angle,  # steering angle
                            ])
    # print("vehicle state:", vehicle_state)

    return vehicle_state

def get_toe_in(self, wheel_state_global): 
    # Wheel normal expressed in global frame
    wheel_normal = wheel_state_global.rot.GetYaxis()
    # Terrain normal at wheel location expressed in global frame
    Z_dir = self.terrain.GetNormal(wheel_state_global.pos)
    # Longitudinal (heading) and lateral directions, in the terrain plane
    wheel_normal_np = np.array([wheel_normal.x, wheel_normal.y, wheel_normal.z])

    #Vcross cannot be found in pychrono. So I'm converting ChVectorD to numpy array and using np.cross
    Z_dir_np = np.array([Z_dir.x, Z_dir.y, Z_dir.z])
    X_dir_np = np.cross(wheel_normal_np, Z_dir_np) 
    X_dir = chrono.ChVectorD(X_dir_np[0], X_dir_np[1], X_dir_np[2])
    X_dir.Normalize()
    X_dir_np = np.array([X_dir.x, X_dir.y, X_dir.z])   
    Y_dir_np = np.cross(Z_dir_np, X_dir_np)
    Y_dir = chrono.ChVectorD(Y_dir_np[0], Y_dir_np[1], Y_dir_np[2])
    rot = chrono.ChMatrix33D()
    rot.Set_A_axis(X_dir, Y_dir, Z_dir)
    tire_csys = chrono.ChCoordsysD(wheel_state_global.pos, rot.Get_A_quaternion()) 

    # Express wheel normal in tire frame
    n = tire_csys.TransformDirectionParentToLocal(wheel_normal)
    # print("n",n.x, n.y, n.z)

    # Wheel normal in the vehicle frame
    n_v = self.my_hmmwv.GetVehicle().GetTransform().TransformDirectionLocalToParent(wheel_normal)

    # Toe-in
    # toe_in = math.atan2(n_v.x, n_v.y)
    toe_in = math.atan2(n.x, n.y)

    return toe_in

def reset_config(self, vehicle_params):
    print("self.config.MASS",self.config.MASS)
    self.config.LENGTH      = vehicle_params.LENGTH
    self.config.WIDTH       = vehicle_params.WIDTH
    self.config.LR          = vehicle_params.LR
    self.config.LF          = vehicle_params.LF
    self.config.WB          = vehicle_params.WB
    self.config.MIN_STEER   = vehicle_params.MIN_STEER
    self.config.MAX_STEER   = vehicle_params.MAX_STEER
    self.config.MAX_STEER_V = vehicle_params.MAX_STEER_V
    self.config.MAX_SPEED   = vehicle_params.MAX_SPEED
    self.config.MIN_SPEED   = vehicle_params.MIN_SPEED
    self.config.MAX_ACCEL   = vehicle_params.MAX_ACCEL
    self.config.MAX_DECEL   = vehicle_params.MAX_DECEL
    self.config.MASS        = vehicle_params.MASS
    print("self.config.MASS",self.config.MASS)

class VehicleParameters:
    def __init__(self, vehicle):
        self.LENGTH: float = 4.298  # Length of the vehicle [m]
        # WIDTH: float = 1.674  # Width of the vehicle [m]
        # LR: float = 1.50876
        # LF: float = 0.88392
        # WB: float = 0.88392 + 1.50876  # Wheelbase [m]
        # MIN_STEER: float = -0.4189  # maximum steering angle [rad]
        # MAX_STEER: float = 0.4189  # maximum steering angle [rad]
        self.MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
        self.MIN_STEER_V: float = 3.2  # maximum steering speed [rad/s]
        self.MAX_SPEED: float = 45.0  # maximum speed [m/s]
        self.MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
        self.MAX_ACCEL: float = 11.5  # maximum acceleration [m/ss]
        self.MAX_DECEL: float = -45.0  # maximum acceleration [m/ss]
        self.MASS = vehicle.GetVehicle().GetMass()
        self.WB   = vehicle.GetVehicle().GetWheelbase()
        self.MIN_STEER = -vehicle.GetVehicle().GetMaxSteeringAngle()
        self.MAX_STEER = +vehicle.GetVehicle().GetMaxSteeringAngle()
        self.WIDTH = vehicle.GetVehicle().GetWheeltrack(0)
        chassisPos = vehicle.GetVehicle().GetChassis().GetPos()
        COMPos = vehicle.GetVehicle().GetChassis().GetCOMFrame().coord.pos
        absPosCOM = COMPos + chassisPos
        fw = vehicle.GetVehicle().GetAxle(0).GetWheels()[1].GetPos()
        tmp = fw - absPosCOM
        self.LF = np.linalg.norm(np.array([tmp.x, tmp.y, tmp.z]))
        self.LR = self.WB - self.LF


# def get_vehicle_parameters(vehicle):
#     params = VehicleParameters()
#     params.MASS = vehicle.GetVehicle().GetMass()
#     params.WB   = vehicle.GetVehicle().GetWheelbase()
#     params.MIN_STEER = -vehicle.GetVehicle().GetMaxSteeringAngle()
#     params.MAX_STEER = +vehicle.GetVehicle().GetMaxSteeringAngle()
#     params.WIDTH = vehicle.GetVehicle().GetWheeltrack(0)
#     chassisPos = vehicle.GetVehicle().GetChassis().GetPos()
#     COMPos = vehicle.GetVehicle().GetChassis().GetCOMFrame().coord.pos
#     absPosCOM = COMPos + chassisPos
#     fw = vehicle.GetVehicle().GetAxle(0).GetWheels()[1].GetPos()
#     tmp = fw - absPosCOM
#     params.LF = np.linalg.norm(np.array([tmp.x, tmp.y, tmp.z]))
    
#     params.LR = params.WB - params.LF 

#     return params       