import os, sys
import time, datetime
import requests
import subprocess

ABSOLUTE_PATH = os.getcwd()
TMP_CLIPS_PATH = os.path.join(ABSOLUTE_PATH, "tmp_clips")
FILENAMES = os.path.join(ABSOLUTE_PATH, "tmp_clips", "filenames.txt")
def save_file(file_url=None, output_dir=None, filenames_file=None):
    
    filename = os.path.split(file_url)[1]
    try:
        response = requests.get(file_url)
        try:
            cur_time = datetime.datetime.now()

            pre_ts_name = filename[:filename.find(".")]
            ts_name = "{}-{}_{}:{}:{}.ts".format(cur_time.month, cur_time.day, cur_time.hour, cur_time.minute, cur_time.second)
            avi_name = "{}-{}_{}:{}:{}.avi".format(cur_time.month, cur_time.day, cur_time.hour, cur_time.minute, cur_time.second)

            videofile = open(os.path.join(output_dir, ts_name), 'wb')
            videofile.write(response.content)
            videofile.close()
            #convert file to avi in background
            
            # subprocess.call(['ffmpeg', '-i', os.path.join(output_dir, ts_name), '-b:v', '4M', '-maxrate', '4M', '-bufsize', '2M', os.path.join(output_dir, avi_name)])
            filenames_file.write("file '{}'\n".format(ts_name))
            print "Successfully saved {} and added it to {}!".format(filename, FILENAMES)
        except:
            print "Saving {} failed!\n{}".format(filename, sys.exc_info()[0])
    except:
        print "Request to {} failed!".format(file_url)

def knit_and_save(filenames_file=None):
    if (filenames_file != None):
	    filenames_file.close()

    filenames_file = open(FILENAMES, 'r')
    #first compress sub-frames into one using ffmpeg
    filename_content = filenames_file.read()
    count = 0
    for i in range(len(filename_content)):
        if filename_content[i] == ":":
            count += 1
        if (count == 2):
            break
    start = filename_content.find("'")+1; end = filename_content.find("'", start) 
    end = i
    filename_start = filename_content.splitlines()[0][start : end] + ".ts"
    print "populating {}".format(filename_start)


    #ffmpeg -f concat -i filenames.txt -c copy video_draft.avi
    destination = os.path.join(ABSOLUTE_PATH, filename_start)
    subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', FILENAMES, '-c', 'copy', destination])

    #tar_dst = os.path.join(ABSOLUTE_PATH, "{}.tar.gz".format(filename_start[:-4]))
    #once the concatenation is done, zip it...
#    subprocess.call(['sudo', 'tar', '-czf', tar_dst, destination])

    #and send it off to mechagodzilla
    # servername = "mechagodzilla.personalrobotics.ri.cmu.edu"
    # username = "rkaufman"
    # server_dst = "/media/data/rkaufman/webcam_footage"
    # subprocess.call(['scp', tar_dst, "{}@{}:{}".format(username, servername, server_dst)])

def download_requests():
    base_url = "https://5ab0d1d85f91f.streamlock.net/Riptydz/cam-rt-bar.stream/"

    ## get first chunk playlist file
    initial_take = base_url + "playlist.m3u8"
    for line in requests.get(initial_take).content.splitlines():
        if line.startswith("chunklist"):
            chunklist_name = line
            print(chunklist_name)
            break


    #query from chunk playlist to collect chunks (each of which is about 9 seconds of film)

    lines_to_query = []
    lines_seen = set()
    prev_time = time.time()
    start_hour = datetime.datetime.now().hour
    filenames_file = open(FILENAMES, "w+")
    i = 0
    while True:
        if (datetime.datetime.now().hour == 23):
            #when hour is past closing we don't want any more stream info
            return
        
        for line in requests.get(base_url + chunklist_name).content.splitlines():
            if (line.startswith("media")):
                lines_to_query.append(line)


        for line in lines_to_query:
            if not (line in lines_seen):
                i+=1
                cur_time = time.time()
                print "New clip took {} seconds to show up".format(cur_time - prev_time)
                prev_time = cur_time
                save_file(
                        file_url=base_url + line, 
                        output_dir=os.path.join(ABSOLUTE_PATH, "tmp_clips"),
                        filenames_file=filenames_file)
                lines_seen.add(line)
        lines_to_query=[]

        #wait 8ish seconds before next query
        time.sleep(8)
        if (datetime.datetime.now().hour - start_hour == 2):
            ## every two hours, knit sub-clips into two hour clip and save it up
            knit_and_save(filenames_file=filenames_file)

            ## delete all tmp_clips
                # rm TMP_CLIPS_PATH/*.ts
            subprocess.call("rm {}".format(os.path.join(TMP_CLIPS_PATH, "*.ts")), shell=True)
            # reset start hour for next two hour block
            start_hour = datetime.datetime.now().hour
            
            # reset filenames.txt file
            filenames_file = open(FILENAMES, "w+")
	
download_requests()
