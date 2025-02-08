import os
import json

def check_json_files(directory, output_file):
    invalid_files = set()


    for filename in os.listdir(directory):

        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)

            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)


                    if 'zoneStates' in data:

                        file_invalid = False


                        for date_time, zone_state in data['zoneStates'].items():

                            if 'zoneKey' in zone_state:
                                expected_name = filename[:-5]
                                actual_zone_key = zone_state['zoneKey']

                                if actual_zone_key != expected_name:
                                    file_invalid = True
                                    break
                            else:
                                file_invalid = True
                                break


                        if file_invalid:
                            invalid_files.add(filename)

                    else:
                        invalid_files.add(filename)

            except json.JSONDecodeError:
                invalid_files.add(filename)
            except Exception as e:
                invalid_files.add(filename)


    with open(output_file, 'w') as output:
        for filename in invalid_files:
            output.write(filename + "\n")


directory = "../Data"
output_file = "invalid_zonekeys.txt"
check_json_files(directory, output_file)
