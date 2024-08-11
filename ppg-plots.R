#? LIBRARIES:
library(tidyverse)  # for data manipulation
library(ggplot2)    # for data visualization
library(ggdark)     # for dark theme


fps <- 29.97*2      # 59.94 Hz

folder <- 'output-data-video/steve-face-hr.mp4-1' # folder with data

#* FUNCTION: to read PPG data
read_ppg <- function(filename, fps) { 
    read_csv(filename) %>%
        mutate(t = (1:n())/fps)
}

#* read RGB data
rgb <-                                
    read_ppg(
        file.path(folder, 'ppg-rgb.csv'),
        fps
    )

#* plot RGB data
ggplot(rgb, aes(x=t)) +
    geom_line(aes(y=r), colour='red') +
    geom_line(aes(y=g), colour='green') +
    geom_line(aes(y=b), colour='blue') +
    dark_theme_bw()

#* read RGB moving average data
rgb_ma <-
    read_ppg(
        file.path(folder, 'ppg-rgb-ma.csv'),
        fps
    )

#* plot RGB moving average data
ggplot(rgb_ma, aes(x=t)) +
    geom_line(aes(y=r), colour='red') +
    geom_line(aes(y=g), colour='green') +
    geom_line(aes(y=b), colour='blue') +
    dark_theme_bw() +
    coord_cartesian(
        xlim = c(30, 40),
        ylim = c(-0.0025, 0.0025),
        expand = TRUE
    )

#* read YUV data
yuv <-
    read_ppg(
        file.path(folder, 'ppg-yuv.csv'),
        fps
    )

#* plot YUV data
ggplot(yuv, aes(x=t)) +
    geom_line(aes(y=y), colour='white') +
    geom_line(aes(y=u), colour='green') +
    geom_line(aes(y=v), colour='magenta') +
    dark_theme_bw()


#* read YUV moving average data
yuv_ma <-
    read_ppg(
        file.path(folder, 'ppg-yuv-ma.csv'),
        fps
    )

#* plot YUV moving average data
ggplot(yuv_ma, aes(x=t)) +
    geom_line(aes(y=y), colour='grey50') +
    geom_line(aes(y=u), colour='green') +
    geom_line(aes(y=v), colour='magenta') +
    dark_theme_bw() +
    coord_cartesian(
        xlim = c(5, 25),
        ylim = c(-0.005, 0.005),
        expand = TRUE
    )



#* read Welch data
ggplot(yuv_ma, aes(x=t)) +
  geom_line(aes(y=v), colour='green') +
  geom_line(aes(y=r), data=rgb_ma, colour='red') +
  dark_theme_bw() +
  coord_cartesian(
    xlim = c(5, 25),
    ylim = c(-0.005, 0.005),
    expand = TRUE
  )


#* read Welch data
welch <-
  read_csv(
    file.path(folder, 'welch.csv'),
  col_names=FALSE
)

#* plot Welch data
ggplot(welch, aes(x=X1, y=X2)) +
  geom_line(colour='black')



#* read ECG data
ecg <- read_delim(
  'ecg/Polar_H10_blah.txt',
  delim=';'
  )

#* plot ECG data
ecg <-
  ecg %>%
  mutate(
    t = `timestamp [ms]`/1000 + 4 + 33/59.94 - 16 - 56/59.94
  )

#* plot ECG data
ggplot(ecg, aes(x=t, y=`ecg [uV]`)) +
  geom_line(colour='red') +
  geom_line(data=rgb_ma, aes(y=g*1e5), colour='green') +
  dark_theme_bw() +
  coord_cartesian(
    xlim = c(20, 30),
    ylim = c(-1000, 2500),
    expand = TRUE
  )