---
orphan: true
---

# MongoDB setup tutorial

`autoplex` and `atomate2` use the MongoDB database framework to store the data output from the calculations. 
The data is stored in a JSON-like format which makes it easier to access through Python for further post-processing.

## MongoDB setup

MongoDB is best run on the back-end. For this, please ask your IT department to install [MongoDB](https://www.mongodb.com/) for you!
There might be cases, where this is not feasible and then you need to run a mongodb on the FRONT-end yourself.
This guide will walk you through the steps that you to do for running a mongodb in front-end.

  * Get mongodb from here: https://www.mongodb.com/try/download/community
  * You can e.g. download "Platform: SUSE 15" and "Package: tgz" and put it on your remote cluster. Check the platform-dependency of your cluster!!!
  * Extract the files from the downloaded archive and add the executables within to your PATH environment variable i.e open the  `~/.bashrc` file and add following line to it.
    ```bash
    export PATH="/path/to/the/mongodb-directory/bin:$PATH"
    ``` 
  * Now, you can start the mongodb database with the following command:
    ```bash
    mongod --bind_ip_all  --dbpath /this_is_the_path_to_your_db --quiet --port 27017
    ```
  * In case you run into errors to start the mongodb on login node, it is mainly due to `mongodb-27017.sock` file permissions. This file is created by mongodb inside /tmp directory. Use the following additional tag in such scenarios (--unixSocketPrefix allows you the change the directory of this file created)
    ```bash
    mongod --bind_ip_all  --dbpath /this_is_the_path_to_your_db --quiet --port 27017 --unixSocketPrefix /new/directory/for/temp/file
    ```
  * You can shut it down with:
    ```bash
    mongod  --shutdown --bind_ip_all  --dbpath /this_is_the_path_to_your_db
    ```
  * Alternatively, you can define a `mongod.conf` file and use that file to start your mongodb database instance. This file looks somewhat like this.  You can read more about the parameters within this file here : https://docs.mongodb.com/manual/administration/configuration/#std-label-base-config
    ```yaml
    processManagement:
       fork: false
       pidFilePath: /dss/dsshome1/00/username/path/to/store/your/pidfile/mongod.pid
    net:
       bindIp: 0.0.0.0
       port: 27017
    storage:
       dbPath: /dss/dsshome1/00/username/path/to/store/your/db/mongo
    systemLog:
       destination: file
       path: /dss/dsshome1/00/username/path/to/mongod.log
       logAppend: true
    storage:
       journal:
          enabled: true
    
    ```

  * You can start the mongodb instance using the `mongod.conf` as follows:

    ```bash
    mongod -f path/to/mongod.conf --quiet --unixSocketPrefix path/to/store/.sock/file
    ```
  * *Note* down the remote cluster login node where you started your mongodb instance.
  * Open another terminal, login to your remote cluster again and ssh to same login node on which your mongodb instance is running.

Note that the next steps of creating databases also have to be done by your IT admin for running MongoDB in backend:

  * For this, open the mongo shell there
    ```bash
    mongosh
    ```
> ℹ️ Note that the [`mongo` shell](https://www.mongodb.com/docs/manual/reference/mongo/) has been deprecated in MongoDB v5.0. The replacement is `mongosh`!

  * You need to add a db called "database_name" (you can choose any name) and also add an admin and a readonly user to this database
  * switch to ```admin``` database and create a user
    ```bash
    use admin
    db.createUser(
      {
        user: "myUserAdmin",
        pwd: "password", 
        roles: [
         { role: "userAdminAnyDatabase", db: "admin" },
          { role: "readWriteAnyDatabase", db: "admin" }
        ]
      }
    )
    ```

  * Add a database for our automation
    ```bash
    use database_name
    ```

  * Test if you are really in this database:
    ```bash
    db
    ```
  * Create a read user for this database:
    ```bash
    db.createUser(
      {
        user: "read",
        pwd:  "password",   // or cleartext password
        roles: [ { role: "read", db: "database_name" } ]
      }
    )
    ```
  * Add again the admin user to this database as well, use the following command
    ```bash
    db.createUser(
      {
        user: "myUserAdmin",
        pwd: "password", 
        roles: [
         { role: "userAdminAnyDatabase", db: "admin" },
          { role: "readWriteAnyDatabase", db: "admin" }
        ]
      }
    )
    ```
  * Close the mongo shell

  ## Alternative MongoDB installation through Homebrew

To install MongoDB through Homebrew, run the following commands in the terminal:
```bash
brew tap mongodb/brew
brew update
brew install mongodb-community
```
Then you can start the server with:
```bash
brew services start mongodb-community
```
Check if the server is running with `brew services list` and it should display something like 
`mongodb-community started uthpala ~/Library/LaunchAgents/homebrew.mxcl.mongodb-community`.


  ## atomate2 configuration

* Create a directory scaffold for atomate2 use `mkdir -p atomate2/{config,logs}`
* Create a new conda environment called atomate2 with Python 3.11 using `conda create -n atomate2 python==3.11`
* Activate your environment : `conda activate atomate2`
* Install atomate2 using : `pip install atomate2`
* Create `atomate2.yaml` file with following content, use `vi ~/atomate2/config/atomate2.yaml`
  ```yaml
  VASP_CMD:  mpirun -np ${SLURM_NTASKS}  /path/to/vasp_std > vasp.out
  VASP_GAMMA_CMD: mpirun -np ${SLURM_NTASKS}  /path/to/vasp_gam > vasp.out
  LOBSTER_CMD: /path/to/your/lobster/binary
  ```

* Create `jobflow.yaml` file with following content, use `vi ~/atomate2/config/jobflow.yaml`
  ```yaml
  JOB_STORE:
    docs_store:
      type: MongoStore
      database: autoplex
      host: local host name
      port: 27017
      username: xxx_admin
      password: xxx
      collection_name: outputs
    additional_stores:
      data:
        type: GridFSStore
        database: autoplex
        host: local host name
        port: 27017
        username: xxx_admin
        password: xxxx
        collection_name: outputs_blobs
  
  ```
* You have to export the location of your files in your ~/.bashrc or ~/.bash_profile
  ```bash
  export ATOMATE2_CONFIG_FILE="/home/username/atomate2/config/atomate2.yaml"
  export JOBFLOW_CONFIG_FILE="/home/username/atomate2/config/jobflow.yaml"
  ```
  

## FireWorks configuration

We will give you a short introduction how to setup [FireWorks](https://github.com/materialsproject/fireworks) to be used with the MongoDB. For [jobflow-remote](https://github.com/Matgenix/jobflow-remote), please have a look [here](jobflowremote.md).

FireWorks can be installed via `pip install fireworks`.

Then follow the next steps:

  * You need to add "my_launchpad.yaml" to your current folder to add information for testing. You need to find out on which login-server you are. ```echo $(hostname)``` and then adapt the following script.
    ```yaml
    host: login_node_name
    port: 27017
    name: database_name
    username: admin
    password: adminpassword
    ssl_ca_file: null
    logdir: null
    strm_lvl: INFO
    user_indices: []
    wf_user_indices: []                  
    ```

  * Once this is done, you can try to use "lpad reset" 
  * Then, you also need a ```my_fworker.yaml``` simlar to this:
    ```yaml
    name: worker
    category: ''
    query: '{}'
    env:
       db_file: /path/to/config/db.json
       vasp_cmd:  srun -N 3 --ntasks=144 /path/to/vasp_std
       lobster_cmd:  /path/to/lobster-5.1.0
       scratch_dir: null
       auto_npar: True
    
    ```

  * And, a ```db.json``` file similar to this:
    ```json
    {
    "host": "login_node_name",
    "port": 27017,
    "database": "database_name",
    "collection": "vasp",
    "admin_user": "adminname",
    "admin_password": "adminpassword",
    "readonly_user": "read",
    "readonly_password": "readpassword",
    "aliases": {}
    }
    
    ```

  * Then, you can use the following job script or something similar to run VASP jobs:
    ```bash
    #!/bin/bash 
    #SBATCH -J vaspjob
    #Output and error
    #SBATCH -o ./%x.%j.out 
    #SBATCH -e ./%x.%j.err 
    #Initial working directory 
    #SBATCH -D ./
    #Notification and type
    #SBATCH --mail-type=END
    #SBATCH --mail-user=your.email@email
    # Wall clock limit: 
    #SBATCH --time=00:30:00
    #SBATCH --no-requeue
    #Setup of execution environment
    #SBATCH --export=NONE 
    #SBATCH --get-user-env
    #SBATCH --nodes=3
    #SBATCH --ntasks=144
    #SBATCH --account=account_name
    #SBATCH --partition=partition_name
    #SBATCH --ear=off 
    
    source /where_is_your_python_environment/bin/activate
    
    module load slurm_setup
    module load vasp/6.1.2
    
    cd /where_do_you_want_to_start_your_calcs/
    
    
    rlaunch   -c /where_do_all_yamls_lie_folder multi 1

    ```
    Please adjust all the scripts according to your needs, setup and cluster requirement!

## Visualize MongoDB-Database

To visualize your database, you can use portforwarding, e.g., ssh remote_cluster -L 8888:localhost:5000. Then open a browser on your local machine to view "http://127.0.0.1:8888/".
