# ClearML

## Useful links

- Setup ClearML server
  [guide](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server)
- ClearML Server configuration
  [guide](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server_config)
- Securing ClearML Server
  [guide](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server_security)
- ClearML Examples
  [repository](https://github.com/allegroai/clearml/tree/master/examples)

## Setup ClearML server

1. See
   [installation guide](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server_linux_mac/)
   for your platform. If you encounter the `elasticserach` error, try to change
   the volume for this service to:

   ```
   - /opt/clearml/elasticsearch/logs:/usr/share/elasticsearch/logs`
   ```

2. Run the `docker-compose` to start the server
3. Initialize `ClearML` client (firstly, you need to install the python
   dependencies):

   ```bash
   clearml-init
   ```

4. Navigate to the `ClearML` web interface. By default, it is available on
   `http://localhost:8080`.
